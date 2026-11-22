"""Tests for the plan-work-verify harness (no model calls, no Docker).

Run from the harness directory:

  uv run --with openai==3.26.1 --with websockets==15.0.1 --with ifcopenshell \\
      python tests/test_harness.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("MODEL_NAME", "test-model")

import serve  # noqa: E402
import taskbridge  # noqa: E402  (imported for its Cancelled type)


class FakeSession:
  """Minimal stand-in for taskbridge.Session."""

  def __init__(self, seconds=21600):
    self.instruction = "Do the engineering task."
    self.workdir = "/app"
    self.limits = {"agent_timeout_sec": seconds}
    self.mcp_servers = []
    self.final_message = ""
    self._seconds = seconds
    self.logs = []

  def time_left(self):
    return self._seconds

  async def log(self, text):
    self.logs.append(text)

  async def usage(self, *args, **kwargs):
    pass


class FakeAgent:
  """Stand-in for serve.Agent recording phases and returning canned text."""

  instances = []
  scripts = {}
  fail_roles = set()
  bash_outputs = []

  def __init__(
      self,
      session,
      role="worker",
      plan="",
      context="",
      phase_seconds=None,
  ):
    self.session = session
    self.role = role
    self.plan = plan
    self.context = context
    self.phase_seconds = phase_seconds
    self.resumes = []
    FakeAgent.instances.append(self)

  async def run(self):
    if self.role in FakeAgent.fail_roles:
      raise RuntimeError(f"{self.role} boom")
    replies = FakeAgent.scripts.setdefault(self.role, [])
    text = replies.pop(0) if replies else f"{self.role} default"
    if text:
      self.session.final_message = text
    return text

  async def resume(self, feedback):
    self.resumes.append(feedback)
    text = f"fixed summary ({len(self.resumes)})"
    self.session.final_message = text
    return text

  async def bash(self, command, timeout_sec=120):
    if "prev_output" in command:
      # The regression guard's snapshot/restore: a deterministic filesystem
      # operation in the task container, not a gate measurement. Pretend it
      # worked so it never consumes a canned gate output.
      return "SNAPSHOT_OK"
    if FakeAgent.bash_outputs:
      return FakeAgent.bash_outputs.pop(0)
    return "exit code 0\nIFC_GATE: SKIP (no .ifc files under output/)"

  @classmethod
  def reset(cls, scripts=None, fail_roles=None, bash_outputs=None):
    cls.instances = []
    cls.scripts = dict(scripts or {})
    cls.fail_roles = set(fail_roles or ())
    cls.bash_outputs = list(bash_outputs or [])

  @classmethod
  def roles(cls):
    return [i.role for i in cls.instances]

  @classmethod
  def first(cls, role):
    return next(i for i in cls.instances if i.role == role)


def check(name, cond, detail=""):
  if not cond:
    raise AssertionError(f"{name}: {detail or 'failed'}")
  print(f"  ok: {name}")


async def test_long_task_pass_first_try():
  print("long task, verifier passes first try")
  FakeAgent.reset(scripts={
      "planner": ["the plan"],
      "worker": ["worker summary"],
      "verifier": ["PASS"],
  })
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(FakeSession(21600))
  finally:
    serve.Agent = real
  check("phases", FakeAgent.roles() == ["planner", "worker", "verifier"],
        FakeAgent.roles())
  check("planner budget", FakeAgent.first("planner").phase_seconds == 2160,
        FakeAgent.first("planner").phase_seconds)
  check("worker budget", FakeAgent.first("worker").phase_seconds == 16200,
        FakeAgent.first("worker").phase_seconds)
  check("worker got plan", FakeAgent.first("worker").plan == "the plan")
  check("verifier got context",
        FakeAgent.first("verifier").context.startswith("worker summary"))
  check("verifier got gate verdict",
        "Deterministic gate results" in FakeAgent.first("verifier").context
        and "IFC gate: no failures reported"
        in FakeAgent.first("verifier").context)
  check("result",
        result == "worker summary\n\nIndependent verification: PASS.",
        repr(result))


async def test_gaps_then_fix_then_pass():
  print("verifier finds gaps, one fix round, then PASS")
  FakeAgent.reset(scripts={
      "planner": ["the plan"],
      "worker": ["worker summary"],
      "verifier": ["1. missing units\n2. wrong path", "PASS"],
  })
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(FakeSession(21600))
  finally:
    serve.Agent = real
  check("phases",
        FakeAgent.roles() == ["planner", "worker", "verifier", "verifier"],
        FakeAgent.roles())
  worker = FakeAgent.first("worker")
  check("one fix round", len(worker.resumes) == 1, worker.resumes)
  check("gaps passed to worker", "missing units" in worker.resumes[0])
  check("result",
        result == "fixed summary (1)\n\nIndependent verification: PASS.",
        repr(result))


async def test_short_task_skips_plan_and_verify():
  print("short task skips planning and verification")
  FakeAgent.reset(scripts={"worker": ["worker summary"]})
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(FakeSession(600))
  finally:
    serve.Agent = real
  check("only worker", FakeAgent.roles() == ["worker"], FakeAgent.roles())
  check("result", result == "worker summary", repr(result))


async def test_planner_failure():
  print("planner failure does not fail the task")
  FakeAgent.reset(
      scripts={"worker": ["worker summary"], "verifier": ["PASS"]},
      fail_roles={"planner"},
  )
  session = FakeSession(21600)
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("worker ran without a plan",
        FakeAgent.first("worker").plan == "")
  check("failure logged",
        any("plan phase failed" in t for t in session.logs),
        session.logs)
  check("result has PASS note", result.endswith("PASS."), repr(result))


async def test_verifier_failure():
  print("verifier failure leaves the summary unverified, not failed")
  FakeAgent.reset(
      scripts={"planner": ["p"], "worker": ["worker summary"]},
      fail_roles={"verifier"},
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(FakeSession(21600))
  finally:
    serve.Agent = real
  check("no fix round", len(FakeAgent.first("worker").resumes) == 0)
  check("result", result == "worker summary", repr(result))


async def test_max_fix_rounds():
  print("fix rounds stop at MAX_FIX_ROUNDS")
  FakeAgent.reset(scripts={
      "planner": ["p"],
      "worker": ["worker summary"],
      "verifier": ["gap1", "gap2", "gap3"],
  })
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(FakeSession(21600))
  finally:
    serve.Agent = real
  check("three verifies",
        FakeAgent.roles().count("verifier") == 3, FakeAgent.roles())
  check("two fix rounds",
        len(FakeAgent.first("worker").resumes) == 2)
  check("result", result == "fixed summary (2)", repr(result))


def test_verdict():
  print("verdict parsing")
  check("PASS", serve._verdict("PASS") == (True, ""))
  check("pass lowercase", serve._verdict("pass\n") == (True, ""))
  check("PASS with note", serve._verdict("PASS: all good") == (True, ""))
  check("gaps", serve._verdict("1. gap") == (False, "1. gap"))
  check("empty", serve._verdict("") == (False, ""))
  check("PASSED is not PASS",
        serve._verdict("PASSED everything")[0] is False)


def test_agent_roles():
  print("real Agent role setup (no network)")
  session = FakeSession(21600)
  planner = serve.Agent(session, role="planner")
  worker = serve.Agent(session, role="worker")
  verifier = serve.Agent(session, role="verifier")
  names = lambda a: {t["function"]["name"] for t in a.tools}  # noqa: E731
  check("planner tools", names(planner) == {"bash", "read_file"},
        names(planner))
  check("verifier tools", names(verifier) == {"bash", "read_file"},
        names(verifier))
  check("worker tools",
        names(worker) == {"bash", "read_file", "write_file", "edit_file"},
        names(worker))
  check("planner prompt mentions plan",
        "plan" in planner.system_prompt("").lower())
  check("verifier prompt mentions PASS",
        "PASS" in verifier.system_prompt(""))
  check("worker prompt has evidence ledger",
        "/tmp/evidence.md" in worker.system_prompt(""))
  check("worker prompt has workdir",
        "/app" in worker.system_prompt(""))
  check("verifier message",
        "Engineer's summary" in verifier.initial_user_message())


def test_phase_time_left():
  print("phase time budget")
  session = FakeSession(21600)
  agent = serve.Agent(session, role="planner", phase_seconds=100)
  left = agent.time_left()
  check("phase caps time", 0 < left <= 100, left)
  agent2 = serve.Agent(session, role="worker")
  check("no phase, task time", agent2.time_left() == 21600)


async def test_resume_clears_phase_deadline():
  print("resume clears the work phase deadline")
  session = FakeSession(21600)
  agent = serve.Agent(session, role="worker", phase_seconds=100)
  agent.messages = [{"role": "user", "content": "x"}]
  calls = []

  async def fake_loop():
    calls.append(agent.phase_deadline)
    return "done"

  agent.loop = fake_loop
  result = await agent.resume("fix this")
  check("resume returns loop result", result == "done")
  check("deadline cleared in loop", calls == [None], calls)
  check("feedback appended",
        agent.messages[-1] == {"role": "user", "content": "fix this"})


async def test_ifc_gate():
  print("IFC gate parsing")
  FakeAgent.reset(bash_outputs=[
      "exit code 0\nIFC_GATE: SKIP (no .ifc files under output/)",
      "exit code 0\nIFC output/model.ifc: 0 EXPRESS violations\n"
      "IFC_GATE: PASS (1 file(s), 0 EXPRESS violations)",
      "exit code 0\nIFC output/model.ifc: 3 EXPRESS violations\n"
      "  IfcDotProduct... != 0.0\nIFC_GATE: FAIL",
      "exit code 127\npython3: command not found",
  ])
  session = FakeSession(21600)
  worker = FakeAgent(session, role="worker")
  check("skip, no files", (await serve._ifc_gate(worker))[0] == "")
  check("pass", (await serve._ifc_gate(worker))[0] == "")
  gaps, scores = await serve._ifc_gate(worker)
  check("violations returned", "3 EXPRESS violations" in gaps, gaps)
  check("scores count violations",
        scores == {"ifc_ok": False, "violations": 3}, scores)
  check("tool error is inconclusive",
        (await serve._ifc_gate(worker))[0] == "")

  class RaisingWorker:
    def __init__(self, session):
      self.session = session

    async def bash(self, command, timeout_sec=120):
      raise RuntimeError("boom")

  check("bash raising is inconclusive",
        (await serve._ifc_gate(RaisingWorker(session)))[0] == "")
  check("gate script compiles",
        compile(serve.IFC_GATE_SCRIPT, "<ifc-gate>", "exec") is not None)
  check("gate command is a heredoc",
        serve.IFC_GATE_COMMAND.startswith("python3 - <<'PYEOF'\n"))


async def test_unit_gate():
  print("unit gate parsing")
  FakeAgent.reset(bash_outputs=[
      "exit code 0\nUNIT_GATE: SKIP (no .ifc files under output/)",
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)",
      "exit code 0\nUNIT output/model.ifc: scale x1000.000,"
      " millimetre unit declared: False\n"
      "UNIT FAIL: output/model.ifc reads 1000x too large — the model has no"
      " millimetre unit declaration\nUNIT_GATE: FAIL",
      "exit code 127\npython3: command not found",
  ])
  session = FakeSession(21600)
  worker = FakeAgent(session, role="worker")
  check("skip, no files", (await serve._unit_gate(worker))[0] == "")
  gaps, scores = await serve._unit_gate(worker)
  check("pass", gaps == "")
  check("pass scores",
        scores == {"unit_ok": True, "scale": 1.0}, scores)
  gaps, scores = await serve._unit_gate(worker)
  check("1000x returned", "1000x too large" in gaps, gaps)
  check("fail scores",
        scores == {"unit_ok": False, "scale": 1000.0}, scores)
  check("tool error is inconclusive",
        (await serve._unit_gate(worker))[0] == "")
  check("gate script compiles",
        compile(serve.UNIT_GATE_SCRIPT, "<unit-gate>", "exec") is not None)
  check("gate command is a heredoc",
        serve.UNIT_GATE_COMMAND.startswith("python3 - <<'PYEOF'\n"))


def _run_unit_gate(workdir):
  import subprocess
  script = os.path.join(workdir, "unit_gate.py")
  with open(script, "w") as f:
    f.write(serve.UNIT_GATE_SCRIPT)
  r = subprocess.run([sys.executable, script], cwd=workdir,
                     capture_output=True, text=True)
  return r.stdout + r.stderr


def test_unit_gate_real_ifc():
  print("unit gate on real IFC files")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import shutil
  import tempfile

  base = os.path.join(
      os.path.dirname(os.path.abspath(__file__)), "fixtures")
  with tempfile.TemporaryDirectory() as tmp:
    # Run 9's model has no unit declaration: the grader reads it 1000x
    # too large. The unit gate must fail it.
    workdir = os.path.join(tmp, "run9")
    shutil.copytree(os.path.join(base, "run9-model"), workdir)
    out = _run_unit_gate(workdir)
    check("run-9 model scale is 1000x", "scale x1000.000" in out, out)
    check("run-9 model fails", "UNIT_GATE: FAIL" in out, out)
    # Run 8's model has the declaration: scale x1.0, passes.
    workdir = os.path.join(tmp, "run8")
    shutil.copytree(os.path.join(base, "run8-model"), workdir)
    out = _run_unit_gate(workdir)
    check("run-8 model scale is 1x", "scale x1.000" in out, out)
    check("run-8 model passes", "UNIT_GATE: PASS" in out, out)


def test_regression_guard():
  print("regression guard score comparison")
  base = {
      "unit_ok": True, "ifc_ok": True, "frame_ok": True,
      "clashes": 1, "max_components": 5, "rows_below_80": 0,
  }
  check("same scores are not a regression",
        not serve._is_regression(base, dict(base)))
  check("clashes up is a regression",
        serve._is_regression(base, {**base, "clashes": 10}))
  check("clashes down is not a regression",
        not serve._is_regression(base, {**base, "clashes": 0}))
  check("unit flipped to FAIL is a regression",
        serve._is_regression(base, {**base, "unit_ok": False}))
  check("frame flipped to FAIL is a regression",
        serve._is_regression(base, {**base, "frame_ok": False}))
  check("components up without clash cut is a regression",
        serve._is_regression(base, {**base, "max_components": 8}))
  check("components up with clash cut is not a regression",
        not serve._is_regression(
            base, {**base, "max_components": 8, "clashes": 0}))
  check("inconclusive scores give no vector",
        serve._gate_score_vector({}, {}, {}, {}) is None)
  check("score vector built",
        serve._gate_score_vector(
            {"unit_ok": True, "scale": 1.0},
            {"ifc_ok": True, "violations": 0},
            {"clashes": 10, "frame_ok": True, "frame_outside": 0},
            {"components": {"tray": 5}, "rows_below_80": 2})
        == {"unit_ok": True, "ifc_ok": True, "frame_ok": True,
            "clashes": 10, "max_components": 5, "rows_below_80": 2})


async def test_gate_blocks_pass_then_fix():
  print("IFC gate turns a verifier PASS into a fix round")
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS"],
      },
      bash_outputs=[
          # Round 1: unit passes, IFC fails (not a routing task, so the
          # separation and connectivity gates do not run).
          "exit code 0\nUNIT output/model.ifc: scale x1.000,"
          " millimetre unit declared: True\n"
          "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)",
          "exit code 0\nIFC output/model.ifc: 1 EXPRESS violations\n"
          "  bad extrusion\nIFC_GATE: FAIL",
          # Round 2: both pass.
          "exit code 0\nUNIT output/model.ifc: scale x1.000,"
          " millimetre unit declared: True\n"
          "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)",
          "exit code 0\nIFC output/model.ifc: 0 EXPRESS violations\n"
          "IFC_GATE: PASS (1 file(s), 0 EXPRESS violations)",
      ],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(FakeSession(21600))
  finally:
    serve.Agent = real
  worker = FakeAgent.first("worker")
  check("one gate fix round", len(worker.resumes) == 1, worker.resumes)
  check("gate text passed to worker",
        "EXPRESS violations" in worker.resumes[0], worker.resumes[0])
  check("result verified",
        result.endswith("Independent verification: PASS."), repr(result))


def test_prompts():
  print("worker/verifier prompt rules and IFC guide")
  session = FakeSession(21600)
  session.instruction = "Build an IFC model of the data hall."
  worker = serve.Agent(session, role="worker")
  prompt = worker.system_prompt("")
  check("worker hard rule",
        "exact check with the exact flags" in prompt)
  check("no pip install", "do not try pip install" in prompt)
  check("IFC guide present",
        "express_rules=True" in prompt and "WR1" in prompt)
  check("IFC snippet present",
        "ifcopenshell.validate.validate(model, _log, express_rules=True)"
        in prompt)
  session2 = FakeSession(21600)
  session2.instruction = "Design a steel frame."
  worker2 = serve.Agent(session2, role="worker")
  check("no IFC guide without IFC",
        "express_rules=True" not in worker2.system_prompt(""))
  verifier = serve.Agent(session, role="verifier")
  vprompt = verifier.system_prompt("")
  check("verifier re-run rule",
        "Re-run the task's stated validation yourself" in vprompt)
  check("verifier distrusts self-check",
        "not independent evidence" in vprompt)


async def test_conn_gate():
  print("connectivity gate parsing")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  worker = FakeAgent(session, role="worker")
  FakeAgent.reset(bash_outputs=[
      "exit code 0\nCONN_GATE: SKIP (no .ifc files under output/)",
      "exit code 0\nCONN tray: 5 segment(s)\nCONN_GATE: PASS",
      "exit code 0\nCONN tray: 1 segment(s), 23000 mm total\n"
      "CONN FAIL: tray: only 1 segment (23000 mm)\nCONN_GATE: FAIL",
      "exit code 127\npython3: command not found",
  ])
  check("skip, no files", (await serve._connectivity_gate(worker))[0] == "")
  check("pass", (await serve._connectivity_gate(worker))[0] == "")
  gaps, scores = await serve._connectivity_gate(worker)
  check("violations returned", "only 1 segment" in gaps, gaps)
  check("scores carry components and rows",
        scores == {"components": {}, "rows_below_80": 0}, scores)
  check("tool error is inconclusive",
        (await serve._connectivity_gate(worker))[0] == "")

  session2 = FakeSession(21600)  # default instruction: not a routing task
  worker2 = FakeAgent(session2, role="worker")
  FakeAgent.reset(bash_outputs=["exit code 0\nCONN_GATE: FAIL"])
  check("non-routing task skips the gate",
        (await serve._connectivity_gate(worker2))[0] == "")
  check("gate did not run a bash call",
        FakeAgent.bash_outputs == ["exit code 0\nCONN_GATE: FAIL"])


async def test_conn_gate_blocks_pass_then_fix():
  print("connectivity gate turns a verifier PASS into a fix round")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS"],
      },
      bash_outputs=[
          # Round 1: unit, IFC, SEP, CONN — consumed in that order.
          "exit code 0\nUNIT_GATE: SKIP (no .ifc files under output/)",
          "exit code 0\nIFC_GATE: SKIP (no .ifc files under output/)",
          "exit code 0\nSEP_GATE: SKIP (no .ifc files under output/)",
          "exit code 0\nCONN tray: 1 segment(s), 23000 mm total\n"
          "CONN FAIL: tray: only 1 segment (23000 mm)\nCONN_GATE: FAIL",
          # Round 2: all pass.
          "exit code 0\nUNIT_GATE: SKIP (no .ifc files under output/)",
          "exit code 0\nIFC_GATE: SKIP (no .ifc files under output/)",
          "exit code 0\nSEP_GATE: SKIP (no .ifc files under output/)",
          "exit code 0\nCONN tray: 5 segment(s)\nCONN_GATE: PASS",
      ],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  worker = FakeAgent.first("worker")
  check("one conn gate fix round", len(worker.resumes) == 1, worker.resumes)
  check("gate text passed to worker",
        "only 1 segment" in worker.resumes[0], worker.resumes[0])
  check("result verified",
        result.endswith("Independent verification: PASS."), repr(result))


async def test_gates_run_in_parallel_and_merge():
  print("SEP and CONN gates both fail in one round, both reach the fix")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS"],
      },
      bash_outputs=[
          # Round 1: unit and IFC pass, SEP and CONN both fail — consumed
          # in unit, IFC, SEP, CONN order by asyncio.gather.
          "exit code 0\nUNIT output/model.ifc: scale x1.000,"
          " millimetre unit declared: True\n"
          "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)",
          "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)",
          "exit code 0\nSEP collisions: 75\n"
          "SEP FAIL: 75 clashes (boxes overlapping > 2 mm on all axes)\n"
          "SEP_GATE: FAIL",
          "exit code 0\nCONN tray: 1 segment(s), 23000 mm total\n"
          "CONN FAIL: tray: only 1 segment (23000 mm)\nCONN_GATE: FAIL",
          # Round 2: all four gates pass.
          "exit code 0\nUNIT output/model.ifc: scale x1.000,"
          " millimetre unit declared: True\n"
          "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)",
          "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)",
          "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS",
          "exit code 0\nCONN tray: 5 segment(s)\nCONN_GATE: PASS",
      ],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  worker = FakeAgent.first("worker")
  check("one merged fix round", len(worker.resumes) == 1, worker.resumes)
  check("SEP gaps reached the worker",
        "75 clashes" in worker.resumes[0], worker.resumes[0])
  check("CONN gaps reached the worker",
        "only 1 segment" in worker.resumes[0], worker.resumes[0])
  check("fix prompt demands gate numbers",
        "quote the gate numbers" in worker.resumes[0], worker.resumes[0])
  check("verifier saw both gate failures",
        "Separation gate: FAIL" in FakeAgent.first("verifier").context
        and "Connectivity gate: FAIL"
        in FakeAgent.first("verifier").context)
  check("result verified",
        result.endswith("Independent verification: PASS."), repr(result))
  check("no gate verdict appended when clean",
        "Deterministic gates:" not in result, repr(result))


async def test_gate_verdict_appended_when_rounds_run_out():
  print("gate verdict is appended to the summary when rounds run out")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS", "PASS"],
      },
      bash_outputs=[
          "exit code 0\nUNIT output/model.ifc: scale x1.000,"
          " millimetre unit declared: True\n"
          "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)",
          "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)",
          "exit code 0\nSEP collisions: 75\n"
          "SEP FAIL: 75 clashes (boxes overlapping > 2 mm on all axes)\n"
          "SEP_GATE: FAIL",
          "exit code 0\nCONN_GATE: PASS",
      ] * 3,
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("two fix rounds used",
        len(FakeAgent.first("worker").resumes) == 2,
        FakeAgent.first("worker").resumes)
  check("gate verdict appended",
        "Deterministic gates:" in result, repr(result))
  check("SEP failure named in the summary",
        "Separation gate: FAIL" in result, repr(result))
  check("not marked verified",
        "Independent verification: PASS." not in result, repr(result))


async def test_regression_guard_restores_worse_fix():
  print("a fix round that makes clashes worse is rolled back")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"
  conn_pass = "exit code 0\nCONN_GATE: PASS"

  def sep_fail(n):
    return (
        f"exit code 0\nSEP collisions: {n}\n"
        f"SEP FAIL: {n} clashes (boxes overlapping > 2 mm on all axes)\n"
        "SEP_GATE: FAIL")

  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS", "PASS"],
      },
      bash_outputs=[
          # Round 1: 75 clashes.
          unit_pass, ifc_pass, sep_fail(75), conn_pass,
          # Round 2: the fix made it worse (100 clashes) — the guard
          # restores the snapshot and re-runs the gates (75 again).
          unit_pass, ifc_pass, sep_fail(100), conn_pass,
          unit_pass, ifc_pass, sep_fail(75), conn_pass,
          # Round 3: still 75 clashes, rounds run out.
          unit_pass, ifc_pass, sep_fail(75), conn_pass,
      ],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  worker = FakeAgent.first("worker")
  check("two fix rounds used", len(worker.resumes) == 2, worker.resumes)
  check("regression note given to the worker",
        "made the model worse" in worker.resumes[1], worker.resumes[1])
  check("restored gaps given, not the worse ones",
        "75 clashes" in worker.resumes[1]
        and "100 clashes" not in worker.resumes[1], worker.resumes[1])
  check("gate verdict appended",
        "Deterministic gates:" in result, repr(result))
  check("not marked verified",
        "Independent verification: PASS." not in result, repr(result))


def test_merge_gate_gaps():
  print("gate gap merging and clipping")
  check("empty merge", serve._merge_gate_gaps("", "", "") == "")
  check("one gate kept", serve._merge_gate_gaps("", "SEP FAIL: 3", "") == "SEP FAIL: 3")
  merged = serve._merge_gate_gaps("IFC FAIL: x", "SEP FAIL: y", "CONN FAIL: z")
  check("all gates kept in order",
        merged.index("IFC") < merged.index("SEP") < merged.index("CONN"),
        merged)
  long = "x" * 5000
  check("total capped near 4000",
        len(serve._merge_gate_gaps(long, long, long)) <= 4100)
  verdict = serve._gate_verdict_lines("", "IFC FAIL: 3", "", "", True)
  check("verdict names the failing gate",
        "IFC gate: FAIL" in verdict and "Separation gate: no failures reported"
        in verdict, verdict)
  check("unit gate reported first",
        verdict.index("Unit gate:") < verdict.index("IFC gate:"), verdict)
  verdict = serve._gate_verdict_lines("", "", "", "", False)
  check("non-routing gates not applicable",
        "not applicable" in verdict, verdict)


def test_routing_prompts():
  print("routing prompt guides and verifier checklist")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water in the data hall."
  worker = serve.Agent(session, role="worker")
  prompt = worker.system_prompt("")
  check("routing guide present", "one header per row" in prompt)
  check("busway class rule", "never IfcElectricDistributionBoard" in prompt)
  check("routing.json contract", "output/routing.json" in prompt)
  check("PNG reader present", "def read_png" in prompt)
  check("butt-joint rule", "butt joints" in prompt)
  check("profile orientation rule", "150 mm" in prompt)
  check("drops go down", "goes down to the CRAH top" in prompt)
  check("tier rule", "separate tiers" in prompt)
  verifier = serve.Agent(session, role="verifier")
  vprompt = verifier.system_prompt("")
  check("verifier connectivity checklist",
        "recompute the route geometry" in vprompt)
  check("verifier quotes numbers", "a PASS that does not quote them" in vprompt)
  check("verifier distrusts empty clash checks",
        "did not actually measure anything" in vprompt)
  check("verifier gets the IFC snippet",
        "ifcopenshell.validate.validate(model, _log, express_rules=True)"
        in vprompt)
  check("verifier gets the routing guide", "one header per row" in vprompt)
  session2 = FakeSession(21600)
  session2.instruction = "Design a steel frame."
  worker2 = serve.Agent(session2, role="worker")
  check("no routing guide without routing",
        "one header per row" not in worker2.system_prompt(""))
  check("no PNG reader without routing",
        "def read_png" not in worker2.system_prompt(""))


def _build_test_ifc(path, segments):
  """Builds a minimal IFC4 file with the given route segments.

  segments: list of (cls, name, w, d, h, x, y, z) boxes in the agent's
  style (long profile, short extrusion, unrotated placement), with an
  optional 8th element (objecttype) for the entity's ObjectType.
  """
  import ifcopenshell

  model = ifcopenshell.file(schema="IFC4")
  units = model.create_entity(
      'IfcUnitAssignment', Units=[
          model.create_entity('IfcSIUnit', UnitType='LENGTHUNIT', Prefix='MILLI', Name='METRE'),
      ])
  project = model.create_entity('IfcProject', GlobalId=ifcopenshell.guid.new(),
                                Name='Test', UnitsInContext=units)
  storey = model.create_entity('IfcBuildingStorey', GlobalId=ifcopenshell.guid.new(),
                               Name='GF', Elevation=0.0)
  axis_world = model.create_entity(
      'IfcAxis2Placement3D',
      Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0]))
  context = model.create_entity(
      'IfcGeometricRepresentationContext', ContextIdentifier='Model',
      ContextType='Model', CoordinateSpaceDimension=3, Precision=0.01,
      WorldCoordinateSystem=axis_world)
  for seg in segments:
    cls, name, w, d, h, x, y, z = seg[:8]
    objecttype = seg[8] if len(seg) > 8 else None
    half_w, half_d = w / 2.0, d / 2.0
    pt = model.create_entity('IfcCartesianPoint', Coordinates=[float(x), float(y), float(z)])
    placement = model.create_entity(
        'IfcLocalPlacement',
        RelativePlacement=model.create_entity('IfcAxis2Placement3D', Location=pt))
    profile = model.create_entity(
        'IfcArbitraryClosedProfileDef', ProfileType='AREA',
        OuterCurve=model.create_entity('IfcPolyLine', Points=[
            model.create_entity('IfcCartesianPoint', Coordinates=[-half_w, -half_d]),
            model.create_entity('IfcCartesianPoint', Coordinates=[half_w, -half_d]),
            model.create_entity('IfcCartesianPoint', Coordinates=[half_w, half_d]),
            model.create_entity('IfcCartesianPoint', Coordinates=[-half_w, half_d]),
            model.create_entity('IfcCartesianPoint', Coordinates=[-half_w, -half_d]),
        ]))
    solid = model.create_entity(
        'IfcExtrudedAreaSolid', SweptArea=profile,
        Position=model.create_entity(
            'IfcAxis2Placement3D',
            Location=model.create_entity('IfcCartesianPoint', Coordinates=[0.0, 0.0, 0.0])),
        ExtrudedDirection=model.create_entity('IfcDirection', DirectionRatios=[0.0, 0.0, 1.0]),
        Depth=float(h))
    shape = model.create_entity(
        'IfcShapeRepresentation', ContextOfItems=context,
        RepresentationIdentifier='Body', RepresentationType='SweptSolid', Items=[solid])
    product = model.create_entity(
        cls, GlobalId=ifcopenshell.guid.new(), Name=name,
        ObjectType=objecttype,
        ObjectPlacement=placement,
        Representation=model.create_entity('IfcProductDefinitionShape', Representations=[shape]))
    model.create_entity(
        'IfcRelContainedInSpatialStructure', GlobalId=ifcopenshell.guid.new(),
        RelatingStructure=storey, RelatedElements=[product])
  model.write(path)


def test_conn_gate_real_ifc():
  print("connectivity gate on real IFC files")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import json
  import subprocess
  import tempfile

  def run_gate(workdir):
    script = os.path.join(workdir, "conn_gate.py")
    with open(script, "w") as f:
      f.write(serve.CONN_GATE_SCRIPT)
    r = subprocess.run([sys.executable, script], cwd=workdir,
                       capture_output=True, text=True)
    return r.stdout + r.stderr

  with tempfile.TemporaryDirectory() as tmp:
    # Good: a connected route (feeder + header per service, one CHW drop).
    outdir = os.path.join(tmp, "good", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 1800, 600, 300, 1100, 3360, 2600),
        ('IfcCableCarrierSegment', 'CableTray-2', 600, 1640, 300, 2000, 4180, 2600),
        ('IfcCableCarrierSegment', 'Busway-1', 1800, 600, 300, 1100, 3960, 2600),
        ('IfcCableCarrierSegment', 'Busway-2', 600, 1640, 300, 2000, 4780, 2600),
        ('IfcPipeSegment', 'CHW-1', 1800, 600, 300, 1100, 2760, 2600),
        ('IfcPipeSegment', 'CHW-2', 600, 2240, 300, 2000, 3880, 2600),
        ('IfcPipeSegment', 'CHW-Drop-1', 300, 300, 1600, 2000, 5000, 1000),
    ])
    json.dump({
        'entry': [200, 3360, 0],
        'rows': [{'name': 'A', 'start': [1100, 3360, 0], 'end': [2000, 3360, 0]}],
        'crahs': [[2000, 5000, 0]],
    }, open(os.path.join(outdir, "routing.json"), "w"))
    out = run_gate(os.path.join(tmp, "good"))
    check("good route passes", "CONN_GATE: PASS" in out, out)

    # Run-6 replica: one box per service, busway as a board.
    outdir = os.path.join(tmp, "run6", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray', 23000, 600, 300, 13500, 10100, 2600),
        ('IfcElectricDistributionBoard', 'Busway', 23000, 400, 300, 13500, 10700, 2600),
        ('IfcPipeSegment', 'CHWPipe', 23000, 300, 300, 13500, 7000, 2800),
    ])
    out = run_gate(os.path.join(tmp, "run6"))
    check("run-6 replica fails", "CONN_GATE: FAIL" in out, out)
    check("run-6 replica catches the single box", "only 1 segment" in out, out)
    check("run-6 replica catches the busway class",
          "busway modelled as IfcElectricDistributionBoard" in out, out)

    # Disconnected: two tray segments 1000 mm apart.
    outdir = os.path.join(tmp, "disc", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 11800, 600, 300, 6100, 5000, 2600),
        ('IfcCableCarrierSegment', 'CableTray-2', 12000, 600, 300, 19000, 5000, 2600),
        ('IfcCableCarrierSegment', 'Busway-1', 24800, 600, 300, 12600, 5000, 2600),
        ('IfcPipeSegment', 'CHW-1', 24800, 600, 300, 12600, 5000, 2600),
    ])
    out = run_gate(os.path.join(tmp, "disc"))
    check("disconnected route fails", "CONN_GATE: FAIL" in out, out)
    check("disconnected route caught", "2 disconnected components" in out, out)

    # Butt joints: boxes touch end to end (0 mm overlap), so the centrelines
    # are a half-width apart at each corner — the gate must still see one
    # connected component per service (it measures the solids, not the
    # centrelines). The old endpoint-based gate called this 5 components.
    outdir = os.path.join(tmp, "butt", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 1800, 600, 300, 1100, 3360, 2600),
        ('IfcCableCarrierSegment', 'CableTray-2', 600, 1340, 300, 2000, 4330, 2600),
        ('IfcCableCarrierSegment', 'CableTray-3', 22700, 600, 300, 13650, 5000, 2600),
        ('IfcCableCarrierSegment', 'Busway-1', 2400, 600, 300, 1400, 3960, 3000),
        ('IfcCableCarrierSegment', 'Busway-2', 600, 1340, 300, 2600, 4930, 3000),
        ('IfcCableCarrierSegment', 'Busway-3', 22100, 600, 300, 13950, 5600, 3000),
        ('IfcPipeSegment', 'CHW-1', 3000, 600, 300, 1700, 2760, 3400),
        ('IfcPipeSegment', 'CHW-2', 600, 1340, 300, 3200, 3730, 3400),
        ('IfcPipeSegment', 'CHW-3', 21500, 600, 300, 14250, 4400, 3400),
        ('IfcPipeSegment', 'CHW-Drop-1', 300, 300, 2400, 3200, 4400, 1000),
    ])
    json.dump({
        'entry': [200, 3360, 0],
        'rows': [{'name': 'A', 'start': [2000, 5000, 0], 'end': [25000, 5000, 0]}],
        'crahs': [[3200, 4400, 0]],
    }, open(os.path.join(outdir, "routing.json"), "w"))
    out = run_gate(os.path.join(tmp, "butt"))
    check("butt-joint route passes", "CONN_GATE: PASS" in out, out)
    check("butt joints are one component per service",
          "1 connected component(s)" in out, out)

    # ObjectType fallback: segments with no Name are classified from
    # ObjectType, and a legitimate PDU is not mistaken for a route element.
    outdir = os.path.join(tmp, "objtype", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', None, 1800, 600, 300, 1100, 3360, 2600, 'CableTray'),
        ('IfcCableCarrierSegment', None, 600, 1640, 300, 2000, 4180, 2600, 'CableTray'),
        ('IfcCableCarrierSegment', None, 1800, 600, 300, 1100, 3960, 2600, 'Busway'),
        ('IfcCableCarrierSegment', None, 600, 1640, 300, 2000, 4780, 2600, 'Busway'),
        ('IfcPipeSegment', None, 1800, 600, 300, 1100, 2760, 2600, 'CHW-Supply'),
        ('IfcPipeSegment', None, 600, 2240, 300, 2000, 3880, 2600, 'CHW-Supply'),
        ('IfcPipeSegment', None, 300, 300, 1600, 2000, 5000, 1000, 'CHW-Supply'),
        ('IfcElectricDistributionBoard', 'PDU-1', 600, 400, 300, 5000, 5000, 2600),
    ])
    json.dump({
        'entry': [200, 3360, 0],
        'rows': [{'name': 'A', 'start': [1100, 3360, 0], 'end': [2000, 3360, 0]}],
        'crahs': [[2000, 5000, 0]],
    }, open(os.path.join(outdir, "routing.json"), "w"))
    out = run_gate(os.path.join(tmp, "objtype"))
    check("unnamed segments classified by ObjectType",
          "CONN_GATE: PASS" in out, out)
    check("PDU is not an unclassified route element",
          "without a recognisable name" not in out, out)

    # No CHW segments: the gate must not print a 1e18 mm CRAH distance.
    outdir = os.path.join(tmp, "nochw", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 1800, 600, 300, 1100, 3360, 2600),
        ('IfcCableCarrierSegment', 'Busway-1', 1800, 600, 300, 1100, 3960, 2600),
    ])
    json.dump({
        'entry': [200, 3360, 0],
        'rows': [{'name': 'A', 'start': [1100, 3360, 0], 'end': [2000, 3360, 0]}],
        'crahs': [[2000, 5000, 0]],
    }, open(os.path.join(outdir, "routing.json"), "w"))
    out = run_gate(os.path.join(tmp, "nochw"))
    check("no CHW fails", "CONN_GATE: FAIL" in out, out)
    check("no 1e18 sentinel in the output",
          "1000000000000000000" not in out, out)


def _run_sep_gate(workdir):
  import subprocess
  script = os.path.join(workdir, "sep_gate.py")
  with open(script, "w") as f:
    f.write(serve.SEP_GATE_SCRIPT)
  r = subprocess.run([sys.executable, script], cwd=workdir,
                     capture_output=True, text=True)
  return r.stdout + r.stderr


async def test_sep_gate():
  print("separation gate parsing")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  worker = FakeAgent(session, role="worker")
  FakeAgent.reset(bash_outputs=[
      "exit code 0\nSEP_GATE: SKIP (no .ifc files under output/)",
      "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS",
      "exit code 0\nSEP collisions: 75\n"
      "SEP FAIL: 75 clashes (boxes overlapping > 2 mm on all axes)\nSEP_GATE: FAIL",
      "exit code 127\npython3: command not found",
  ])
  check("skip, no files", (await serve._separation_gate(worker))[0] == "")
  check("pass", (await serve._separation_gate(worker))[0] == "")
  gaps, scores = await serve._separation_gate(worker)
  check("clashes returned", "75 clashes" in gaps, gaps)
  check("scores count clashes",
        scores == {"clashes": 75, "frame_ok": True, "frame_outside": 0},
        scores)
  check("tool error is inconclusive",
        (await serve._separation_gate(worker))[0] == "")

  session2 = FakeSession(21600)  # default instruction: not a routing task
  worker2 = FakeAgent(session2, role="worker")
  FakeAgent.reset(bash_outputs=["exit code 0\nSEP_GATE: FAIL"])
  check("non-routing task skips the gate",
        (await serve._separation_gate(worker2))[0] == "")
  check("gate did not run a bash call",
        FakeAgent.bash_outputs == ["exit code 0\nSEP_GATE: FAIL"])
  check("gate script compiles",
        compile(serve.SEP_GATE_SCRIPT, "<sep-gate>", "exec") is not None)
  check("gate command is a heredoc",
        serve.SEP_GATE_COMMAND.startswith("python3 - <<'PYEOF'\n"))


def test_sep_gate_real_ifc():
  print("separation gate on real IFC files")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import json
  import shutil
  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    # The butt-joint good case from the connectivity test: 0 clashes.
    outdir = os.path.join(tmp, "butt", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 1800, 600, 300, 1100, 3360, 2600),
        ('IfcCableCarrierSegment', 'CableTray-2', 600, 1340, 300, 2000, 4330, 2600),
        ('IfcCableCarrierSegment', 'CableTray-3', 22700, 600, 300, 13650, 5000, 2600),
        ('IfcCableCarrierSegment', 'Busway-1', 2400, 600, 300, 1400, 3960, 3000),
        ('IfcCableCarrierSegment', 'Busway-2', 600, 1340, 300, 2600, 4930, 3000),
        ('IfcCableCarrierSegment', 'Busway-3', 22100, 600, 300, 13950, 5600, 3000),
        ('IfcPipeSegment', 'CHW-1', 3000, 600, 300, 1700, 2760, 3400),
        ('IfcPipeSegment', 'CHW-2', 600, 1340, 300, 3200, 3730, 3400),
        ('IfcPipeSegment', 'CHW-3', 21500, 600, 300, 14250, 4400, 3400),
        ('IfcPipeSegment', 'CHW-Drop-1', 300, 300, 2400, 3200, 4400, 1000),
    ])
    out = _run_sep_gate(os.path.join(tmp, "butt"))
    check("butt-joint route has no clashes", "SEP collisions: 0" in out, out)
    check("butt-joint route passes", "SEP_GATE: PASS" in out, out)

    # Run-6 replica: three separate boxes, no overlap.
    outdir = os.path.join(tmp, "run6", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray', 23000, 600, 300, 13500, 10100, 2600),
        ('IfcElectricDistributionBoard', 'Busway', 23000, 400, 300, 13500, 10700, 2600),
        ('IfcPipeSegment', 'CHWPipe', 23000, 300, 300, 13500, 7000, 2800),
    ])
    out = _run_sep_gate(os.path.join(tmp, "run6"))
    check("run-6 replica has no clashes", "SEP collisions: 0" in out, out)
    check("run-6 replica passes", "SEP_GATE: PASS" in out, out)

    # The real run-7 model: the gate must reproduce the grader's 75
    # clashes exactly, and catch the profile-orientation and drop bugs.
    fixture = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures", "run7-model")
    workdir = os.path.join(tmp, "run7")
    shutil.copytree(fixture, workdir)
    out = _run_sep_gate(workdir)
    check("run-7 model reports 75 clashes", "SEP collisions: 75" in out, out)
    check("run-7 model fails", "SEP_GATE: FAIL" in out, out)
    check("run-7 profile orientation caught",
          "profile is oriented wrong" in out, out)
    check("run-7 upward drop caught", "does not hang from the header" in out, out)
    check("run-7 sprinkler clearance caught",
          "below the sprinklers" in out, out)
    check("run-7 frame is inside the slab",
          "SEP frame: 0 route segment(s) outside the slab footprint" in out,
          out)
    # The real run-9 model: the double-placement bug builds the route at
    # 2x, so 34 segments stick out of the slab; the frame check must catch
    # it, and the drop message must say the gap is vertical.
    fixture = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "fixtures", "run9-model")
    workdir = os.path.join(tmp, "run9")
    shutil.copytree(fixture, workdir)
    out = _run_sep_gate(workdir)
    check("run-9 model reports 10 clashes", "SEP collisions: 10" in out, out)
    check("run-9 frame caught",
          "SEP frame: 34 route segment(s) outside the slab footprint" in out,
          out)
    check("run-9 frame failure named", "FRAME FAIL" in out, out)
    check("run-9 drop gap is vertical",
          "400 mm away to the above" in out, out)


def test_sep_gate_z_band_and_drop():
  print("separation gate tier overlap and drop branch guidance")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    # Two services on the same tier with the same plan footprint: they
    # clash, and the gate must name the shared z band as the cause.
    outdir = os.path.join(tmp, "flat", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 10000, 600, 150, 5000, 5000, 2850),
        ('IfcCableCarrierSegment', 'Busway-1', 10000, 600, 150, 5000, 5000, 2850),
    ])
    out = _run_sep_gate(os.path.join(tmp, "flat"))
    check("flat tiers clash", "SEP collisions: 1" in out, out)
    check("z band named as the cause",
          "tray and busway share z 2850-3000" in out, out)
    check("flat tiers fail", "SEP_GATE: FAIL" in out, out)

    # A drop far from any header: the gate must name the nearest
    # horizontal segment and tell the fix to build a branch.
    outdir = os.path.join(tmp, "drop", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcPipeSegment', 'CHW-1', 10000, 600, 150, 5000, 5000, 2800),
        ('IfcPipeSegment', 'CHW-Drop-1', 300, 300, 2000, 15000, 15000, 800),
    ])
    out = _run_sep_gate(os.path.join(tmp, "drop"))
    check("hanging drop caught", "does not hang from the header" in out, out)
    check("nearest segment named",
          "nearest horizontal segment is CHW-1" in out, out)
    check("branch guidance given",
          "build a horizontal branch" in out, out)
    check("direction given", "to the south" in out, out)


async def main():
  await test_long_task_pass_first_try()
  await test_gaps_then_fix_then_pass()
  await test_short_task_skips_plan_and_verify()
  await test_planner_failure()
  await test_verifier_failure()
  await test_max_fix_rounds()
  test_verdict()
  test_agent_roles()
  test_phase_time_left()
  await test_resume_clears_phase_deadline()
  await test_ifc_gate()
  await test_unit_gate()
  test_unit_gate_real_ifc()
  test_regression_guard()
  await test_gate_blocks_pass_then_fix()
  test_prompts()
  await test_conn_gate()
  await test_conn_gate_blocks_pass_then_fix()
  test_routing_prompts()
  test_conn_gate_real_ifc()
  await test_sep_gate()
  test_sep_gate_real_ifc()
  test_sep_gate_z_band_and_drop()
  test_merge_gate_gaps()
  await test_gates_run_in_parallel_and_merge()
  await test_gate_verdict_appended_when_rounds_run_out()
  await test_regression_guard_restores_worse_fix()
  print("ALL TESTS PASSED")


if __name__ == "__main__":
  asyncio.run(main())
