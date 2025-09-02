import os, json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Literal
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel, Field
from icalendar import Calendar, Event
from ortools.sat.python import cp_model
from dotenv import load_dotenv

load_dotenv()

# =========================
# Config horaires & fermetures
# =========================
AM_START, AM_END = "08:00", "12:00"
PM_START, PM_END = "13:00", "17:00"

# 0=lundi ... 6=dimanche
DOW = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

# >>> FERMETURES (adapter librement)
#   ("mon","AM") = lundi matin fermé
#   ("wed","AM") = mercredi matin fermé
CLOSURES = {("mon","AM"), ("wed","AM")}  # <- demandé

# =========================
# Modèles
# =========================
class Employee(BaseModel):
    first_name: str
    roles: List[str]
    max_hours_per_week: int = 40
    min_rest_hours: int = 11  # heures

class Availability(BaseModel):
    first_name: str
    # {"YYYY-MM-DD": [["HH:MM","HH:MM"], ...]}
    slots_by_date: Dict[str, List[List[str]]]

class Shift(BaseModel):
    date: str   # "YYYY-MM-DD"
    start: str  # "HH:MM"
    end: str
    role: str

# =========================
# Utilitaires
# =========================
def parse_hm(hm: str):
    h, m = hm.split(":")
    return int(h), int(m)

def to_dt(date_str: str, hm: str) -> datetime:
    y, m, d = [int(x) for x in date_str.split("-")]
    hh, mm = parse_hm(hm)
    return datetime(y, m, d, hh, mm)

def shift_duration_min(sh: Shift) -> int:
    return int((to_dt(sh.date, sh.end) - to_dt(sh.date, sh.start)).total_seconds() // 60)

def week_dates(week_monday: str) -> List[str]:
    wd = datetime.fromisoformat(week_monday)
    return [(wd + timedelta(days=i)).date().isoformat() for i in range(7)]

def slot_tuple(kind: Literal["AM","PM"]) -> Tuple[str,str]:
    return (AM_START, AM_END) if kind == "AM" else (PM_START, PM_END)

def add_slot(avmap: Dict[str, List[List[str]]], date: str, start: str, end: str):
    avmap.setdefault(date, [])
    if [start, end] not in avmap[date]:
        avmap[date].append([start, end])

def remove_slot(avmap: Dict[str, List[List[str]]], date: str, start: str, end: str):
    slots = avmap.get(date, [])
    new_slots = [s for s in slots if not (s[0] == start and s[1] == end)]
    if new_slots:
        avmap[date] = new_slots
    elif date in avmap:
        del avmap[date]

def is_open(date_iso: str, kind: Literal["AM","PM"]) -> bool:
    d = datetime.fromisoformat(date_iso)
    wd_name = DOW[d.weekday()]  # "mon","tue",...
    return (wd_name, kind) not in CLOSURES

def is_within_availability(first_name: str, sh: Shift, all_avail: Dict[str, Dict[str, List[List[str]]]]) -> bool:
    a = all_avail.get(first_name, {})
    slots = a.get(sh.date, [])
    sh_s = to_dt(sh.date, sh.start)
    sh_e = to_dt(sh.date, sh.end)
    for s, e in slots:
        s_dt = to_dt(sh.date, s)
        e_dt = to_dt(sh.date, e)
        if s_dt <= sh_s and sh_e <= e_dt:
            return True
    return False

def generate_shifts_for_week(week_monday: str) -> List[Shift]:
    """
    Génère 2 shifts/jour (08:00–12:00 et 13:00–17:00) pour 7 jours,
    en respectant les FERMETURES dans CLOSURES.
    Rôle unique 'service'.
    """
    dates = week_dates(week_monday)
    shifts: List[Shift] = []
    for d in dates:
        if is_open(d, "AM"):
            shifts.append(Shift(date=d, start=AM_START, end=AM_END, role="service"))
        if is_open(d, "PM"):
            shifts.append(Shift(date=d, start=PM_START, end=PM_END, role="service"))
    return shifts

# =========================
# Solveur CP-SAT (OR-Tools)
# =========================
def build_schedule(employees: List[Employee],
                   avail: Dict[str, Dict],
                   shifts: List[Shift]) -> List[Tuple[str, Shift]]:
    emp_names = [e.first_name for e in employees]
    E, S = range(len(emp_names)), range(len(shifts))

    role_map = {e.first_name: set(e.roles) for e in employees}
    can_work = {}
    for si, sh in enumerate(shifts):
        for ei, e in enumerate(employees):
            ok_role = sh.role in role_map[e.first_name]
            ok_dispo = is_within_availability(e.first_name, sh, avail)
            can_work[(ei, si)] = ok_role and ok_dispo

    model = cp_model.CpModel()
    x = {(ei, si): model.NewBoolVar(f"x_{ei}_{si}") for ei in E for si in S}

    # <= 1 personne par shift (et non ==1)
    for si in S:
        model.Add(sum(x[(ei, si)] for ei in E) <= 1)

    # Interdictions (role/dispo)
    for (ei, si), allowed in can_work.items():
        if not allowed:
            model.Add(x[(ei, si)] == 0)

    # Objectif principal : maximiser le nombre de shifts couverts
    total_assigned = model.NewIntVar(0, len(shifts), "total_assigned")
    model.Add(total_assigned == sum(x[(ei, si)] for ei in E for si in S))

    # Objectif secondaire (équité) : on l’ajoute avec une petite pénalité
    mins = [shift_duration_min(shifts[si]) for si in S]
    minutes_by_emp = []
    for ei in E:
        total = sum(x[(ei, si)] * mins[si] for si in S)
        v = model.NewIntVar(0, 7*24*60, f"min_{ei}")
        model.Add(v == total)
        minutes_by_emp.append(v)

    max_h = model.NewIntVar(0, 7*24*60, "max_h")
    min_h = model.NewIntVar(0, 7*24*60, "min_h")
    model.AddMaxEquality(max_h, minutes_by_emp)
    model.AddMinEquality(min_h, minutes_by_emp)
    spread = model.NewIntVar(0, 7*24*60, "spread")
    model.Add(spread == max_h - min_h)

    # Maximize total_assigned en priorité, puis équité (pénaliser spread)
    # (poids 1000 pour être sûr que couvrir un shift vaut plus que l'équité)
    model.Maximize(total_assigned * 1000 - spread)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return []

    assignments = []
    for si in S:
        for ei in E:
            if solver.BooleanValue(x[(ei, si)]):
                assignments.append((emp_names[ei], shifts[si]))
    return assignments

# =========================
# “Base de données” en mémoire
# =========================
DB = {
    "employees": [],          # List[Employee]
    "avail": {},              # { first_name: {date: [[start,end], ...]}}
    "assignments": [],        # List[Tuple(first_name, Shift)]
    "shifts_last": []         # List[Shift]
}

def add_or_replace_employee(e: Employee):
    DB["employees"] = [x for x in DB["employees"] if x.first_name != e.first_name] + [e]

def set_availability(a: Availability):
    DB["avail"][a.first_name] = a.slots_by_date

def build_schedule_api(week_start: str):
    shifts = generate_shifts_for_week(week_start)
    DB["shifts_last"] = shifts
    DB["assignments"] = build_schedule(DB["employees"], DB["avail"], shifts)

# =========================
# Assistant (NLP -> actions)
# =========================
SYSTEM_PROMPT = """Tu es un assistant de planification pour une API.
Ne fais AUCUNE navigation web. Retourne UNIQUEMENT un JSON valide suivant ce schéma:

{
  "employees": [ { "first_name":"Prénom", "roles":["service"], "max_hours_per_week":40, "min_rest_hours":11 } ],
  "availability": [ { "first_name":"Prénom", "slots_by_date": { "YYYY-MM-DD": [["HH:MM","HH:MM"]] } } ],
  "recurring": [ {
     "first_name":"Prénom",
     "week_start":"YYYY-MM-DD",
     "include": { "AM": true/false, "PM": true/false, "days": ["mon","tue","wed","thu","fri","sat","sun"] },
     "except_days": ["fri","sat"]
  } ],
  "swap": [ { "date":"YYYY-MM-DD", "slot":"AM|PM", "out":"Prénom", "in":"Prénom" } ],
  "build_schedule": { "week_start":"YYYY-MM-DD" }
}

Règles horaires: AM=08:00–12:00, PM=13:00–17:00. Fermetures à respecter côté serveur.
Exemples:
- "XX dispo tous les matins sauf vendredi, semaine du 8 sept" =>
  {"recurring":[{"first_name":"XX","week_start":"2025-09-08","include":{"AM":true,"PM":false,"days":["mon","tue","wed","thu","fri","sat","sun"]},"except_days":["fri"]}],"build_schedule":{"week_start":"2025-09-08"}}
- "remplace XX par YY le jeudi après-midi semaine du 8 sept" =>
  {"swap":[{"date":"2025-09-11","slot":"PM","out":"XX","in":"YY"}],"build_schedule":{"week_start":"2025-09-08"}}
- "ajoute ZZ à l'équipe" => {"employees":[{"first_name":"ZZ","roles":["service"]}]}

Toujours retourner du JSON, sans texte autour.
"""

def gpt_extract_actions(text: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":text}],
        temperature=0
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"employees": [], "availability": [], "recurring": [], "swap": [], "build_schedule": None}

def expand_recurring(rule: dict) -> Dict[str, List[List[str]]]:
    """Transforme une règle 'recurring' en slots_by_date, en respectant CLOSURES."""
    week_start = rule["week_start"]
    include = rule.get("include", {})
    include_am = bool(include.get("AM", False))
    include_pm = bool(include.get("PM", False))
    days = include.get("days", DOW)
    except_days = set(rule.get("except_days", []))

    slots: Dict[str, List[List[str]]] = {}
    for i, date_iso in enumerate(week_dates(week_start)):
        dow = DOW[i]
        if dow in except_days or dow not in days:
            continue
        if include_am and is_open(date_iso, "AM"):
            add_slot(slots, date_iso, AM_START, AM_END)
        if include_pm and is_open(date_iso, "PM"):
            add_slot(slots, date_iso, PM_START, PM_END)
    return slots

def apply_actions(actions: dict) -> dict:
    # 1) employees
    for emp in actions.get("employees", []) or []:
        if "roles" not in emp or not emp["roles"]:
            emp["roles"] = ["service"]
        add_or_replace_employee(Employee(**emp))

    # 2) availability (dates explicites)
    for av in actions.get("availability", []) or []:
        DB["avail"][av["first_name"]] = av["slots_by_date"]

    # 3) recurring -> expand to explicit dates (fusion additive)
    for ru in actions.get("recurring", []) or []:
        name = ru["first_name"]
        expanded = expand_recurring(ru)
        cur = DB["avail"].get(name, {})
        for d, slots in expanded.items():
            cur.setdefault(d, [])
            for s in slots:
                if s not in cur[d]:
                    cur[d].append(s)
        DB["avail"][name] = cur

    # 4) swap: rendre indispo 'out' + rendre dispo 'in' sur le créneau
    for sw in actions.get("swap", []) or []:
        date = sw["date"]
        start, end = slot_tuple(sw["slot"])
        out_name, in_name = sw["out"], sw["in"]
        # out
        av_out = DB["avail"].get(out_name, {})
        remove_slot(av_out, date, start, end)
        DB["avail"][out_name] = av_out
        # in
        if is_open(date, sw["slot"]):
            av_in = DB["avail"].get(in_name, {})
            add_slot(av_in, date, start, end)
            DB["avail"][in_name] = av_in

    built = None
    if actions.get("build_schedule"):
        week_start = actions["build_schedule"]["week_start"]
        build_schedule_api(week_start)
        built = {"assigned": len(DB["assignments"]), "week_start": week_start}

    return {"ok": True, "built": built}

def handle_instruction_text(text: str) -> dict:
    actions = gpt_extract_actions(text)
    return {"parsed": actions, "applied": apply_actions(actions)}

# =========================
# FastAPI
# =========================
app = FastAPI(title="AutoPlanner")

@app.post("/add_employee")
def add_employee(e: Employee):
    add_or_replace_employee(e)
    return {"ok": True}

@app.post("/set_availability")
def set_availability_ep(a: Availability):
    set_availability(a)
    return {"ok": True}

class ScheduleRequest(BaseModel):
    week_start: str

@app.post("/build_schedule")
def build_schedule_ep(req: ScheduleRequest):
    build_schedule_api(req.week_start)
    return {"ok": True, "assigned": len(DB["assignments"])}

@app.post("/nlp")
def nlp_apply(req: dict = None):
    text = (req or {}).get("text", "")
    return handle_instruction_text(text)

@app.get("/calendar.ics")
def calendar_ics():
    cal = Calendar()
    cal.add('prodid', '-//AutoPlanner//FR//')
    cal.add('version', '2.0')
    for emp, sh in DB["assignments"]:
        ev = Event()
        ev.add('summary', f"{sh.role}: {emp}")
        ev.add('dtstart', to_dt(sh.date, sh.start))
        ev.add('dtend', to_dt(sh.date, sh.end))
        ev.add('uid', f"{emp}-{sh.date}-{sh.start}-{sh.end}@autoplanner")
        cal.add_component(ev)
    ics = cal.to_ical()
    return Response(content=ics, media_type="text/calendar")

# --- Simulation: upload audio -> STT -> assistant
@app.post("/test/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    suffix = "." + (file.filename.split(".")[-1] if "." in file.filename else "ogg")
    with NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        content = await file.read()
        tmp.write(content); tmp.flush()

        # Transcription OpenAI
        import openai
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("STT_MODEL", "gpt-4o-transcribe")
        with open(tmp.name, "rb") as f:
            trx = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="text",
                temperature=0
            )
        transcript = trx if isinstance(trx, str) else str(trx)

    result = handle_instruction_text(transcript)
    return JSONResponse({"transcript": transcript, "result": result})

# debug
@app.get("/state")
def state():
    return {
        "employees": [e.model_dump() for e in DB["employees"]],
        "avail": DB["avail"],
        "assigned": len(DB["assignments"])
    }