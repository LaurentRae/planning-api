import os, json
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response, JSONResponse
from pydantic import BaseModel
from icalendar import Calendar, Event
from ortools.sat.python import cp_model
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Modèles et utilitaires
# -----------------------------
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

def parse_hm(hm: str):
    h, m = hm.split(":")
    return int(h), int(m)

def to_dt(date_str: str, hm: str) -> datetime:
    y, m, d = [int(x) for x in date_str.split("-")]
    hh, mm = parse_hm(hm)
    return datetime(y, m, d, hh, mm)

def shift_duration_min(sh: Shift) -> int:
    return int((to_dt(sh.date, sh.end) - to_dt(sh.date, sh.start)).total_seconds() // 60)

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
    SAUF lundi matin (fermé) et mercredi après-midi (fermé).
    Rôle unique 'service' pour simplifier.
    """
    wd = datetime.fromisoformat(week_monday)
    shifts: List[Shift] = []
    for i in range(7):
      d = (wd + timedelta(days=i)).date().isoformat()
      weekday = (wd + timedelta(days=i)).weekday()  # 0=lundi ... 2=mercredi
      # matin (sauf lundi)
      if not (weekday == 0):  # pas le lundi matin
          shifts.append(Shift(date=d, start="08:00", end="12:00", role="service"))
      # après-midi (sauf mercredi)
      if not (weekday == 2):  # pas le mercredi après-midi
          shifts.append(Shift(date=d, start="13:00", end="17:00", role="service"))
    return shifts

# -----------------------------
# Solveur CP-SAT (OR-Tools)
# -----------------------------
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

    # 1 personne par shift
    for si in S:
        model.Add(sum(x[(ei, si)] for ei in E) == 1)

    # Interdictions (role/dispo)
    for (ei, si), allowed in can_work.items():
        if not allowed:
            model.Add(x[(ei, si)] == 0)

    # Objectif d’équité: minimise (max_heures - min_heures)
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
    model.Minimize(max_h - min_h)

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

# -----------------------------
# Assistant (texte -> actions)
# -----------------------------
SYSTEM_PROMPT = """Tu es un assistant de planification.
À partir d’un texte FR informel (dicté), extrais des actions JSON strictement.
Schéma:
{
  "employees": [{"first_name":"Prénom","roles":["role1"],"max_hours_per_week":24,"min_rest_hours":11}],
  "availability": [{"first_name":"Prénom","slots_by_date":{"YYYY-MM-DD":[["HH:MM","HH:MM"]]}}],
  "build_schedule": {"week_start":"YYYY-MM-DD"}  // optionnel
}
Toujours retourner du JSON valide sans texte autour.
"""

def gpt_extract_actions(text: str) -> dict:
    import openai
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":text}
        ],
        temperature=0
    )
    content = resp.choices[0].message.content
    try:
        return json.loads(content)
    except Exception:
        return {"employees": [], "availability": [], "build_schedule": None}

def handle_instruction_text(text: str) -> dict:
    actions = gpt_extract_actions(text)
    # appliquer localement (DB en mémoire)
    for emp in actions.get("employees", []):
        add_or_replace_employee(Employee(**emp))
    for av in actions.get("availability", []):
        set_availability(Availability(**av))
    if actions.get("build_schedule"):
        build_schedule_api(actions["build_schedule"]["week_start"])
    return {"ok": True, "applied": actions}

# -----------------------------
# “Base de données” en mémoire
# -----------------------------
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

# -----------------------------
# FastAPI
# -----------------------------
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