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
    # In-memory files for read_file/write_file. output/solution.py is a
    # real (non-stub) submission by default, so the empty-submission
    # guard stays quiet in tests that are not about it; the guard test
    # replaces it with the stub.
    self.files = {
        "output/solution.py": b"# the agent's solution\nprint('real')\n",
    }

  def time_left(self):
    return self._seconds

  async def log(self, text):
    self.logs.append(text)

  async def usage(self, *args, **kwargs):
    pass

  async def read_file(self, path):
    if path in self.files:
      data = self.files[path]
      return data if isinstance(data, bytes) else data.encode()
    raise FileNotFoundError(path)

  async def write_file(self, path, data, append=False):
    data = data if isinstance(data, bytes) else data.encode()
    if append and path in self.files:
      self.files[path] = self.files[path] + data
    else:
      self.files[path] = data
    return len(self.files[path])

  async def exec(self, command, timeout_sec=120):
    # Every command "succeeds": draw_tools.py "runs" (exit 0) but writes
    # no ground_truth.json, so _run_draw_tools returns None unless a test
    # pre-seeds files["ground_truth.json"].
    return {"exit_code": 0, "output": "", "truncated": False,
            "timed_out": False}


class FakeAgent:
  """Stand-in for serve.Agent recording phases and returning canned text."""

  instances = []
  scripts = {}
  fail_roles = set()
  bash_outputs = []
  # The standalone check's canned result: PASS by default, so existing
  # solve() tests are unaffected; a test can set it to a failing run
  # (the run-12 defect: solution.py opens ground_truth.json at runtime).
  standalone_result = "STANDALONE_EXIT:0\nSTANDALONE_IFC:OK"
  standalone_tail = "(no log)"

  def __init__(
      self,
      session,
      role="worker",
      plan="",
      context="",
      phase_seconds=None,
      max_tool_calls=None,
  ):
    self.session = session
    self.role = role
    self.plan = plan
    self.context = context
    self.phase_seconds = phase_seconds
    self.max_tool_calls = max_tool_calls
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
    if "cp output/solution.py" in command and "/tmp/standalone" in command:
      # The standalone check runs solution.py alone in a clean directory;
      # the canned result stands in for the run. Never consumes a gate
      # output.
      return FakeAgent.standalone_result
    if "tail -5 /tmp/standalone" in command:
      # The log tail the standalone check reads when it fails.
      return FakeAgent.standalone_tail
    if FakeAgent.bash_outputs:
      return FakeAgent.bash_outputs.pop(0)
    return "exit code 0\nIFC_GATE: SKIP (no .ifc files under output/)"

  @classmethod
  def reset(cls, scripts=None, fail_roles=None, bash_outputs=None,
            standalone_result=None, standalone_tail=None):
    cls.instances = []
    cls.scripts = dict(scripts or {})
    cls.fail_roles = set(fail_roles or ())
    cls.bash_outputs = list(bash_outputs or [])
    cls.standalone_result = (
        standalone_result if standalone_result is not None
        else "STANDALONE_EXIT:0\nSTANDALONE_IFC:OK")
    cls.standalone_tail = (
        standalone_tail if standalone_tail is not None else "(no log)")

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
  check("planner budget", FakeAgent.first("planner").phase_seconds == 600,
        FakeAgent.first("planner").phase_seconds)
  check("planner tool-call cap",
        FakeAgent.first("planner").max_tool_calls == 10,
        FakeAgent.first("planner").max_tool_calls)
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
      "components": {"tray": 5, "busway": 5, "chw": 5},
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
        serve._is_regression(base, {**base, "max_components": 8,
                                    "components": {"tray": 8, "busway": 5,
                                                   "chw": 5}}))
  # Run 12: a fix cut clashes 92 -> 0 by shortening segments short of
  # their neighbours (tray 1 -> 7, CHW 4 -> 8). The old rule let it
  # through because the clash cut "paid" for the disconnect; a
  # disconnected network scores 0 as surely as a clashing one, so any
  # per-service component rise is a regression now, even with a clash cut.
  check("components up with clash cut is a regression now",
        serve._is_regression(base, {**base, "max_components": 8,
                                    "components": {"tray": 8, "busway": 5,
                                                   "chw": 5},
                                    "clashes": 0}))
  check("one service up is a regression even when others are flat",
        serve._is_regression(base, {**base,
                                    "components": {"tray": 5, "busway": 5,
                                                   "chw": 6}}))
  check("components down is not a regression",
        not serve._is_regression(base, {**base, "max_components": 3,
                                        "components": {"tray": 3,
                                                       "busway": 3,
                                                       "chw": 3}}))
  # Run 15: a fix round closed the last connectivity break (CHW 2 -> 1
  # components, CONN PASS) at the cost of 29 new joint overlaps
  # (SEP 92 -> 113). The old any-gate-worse rule reverted it, leaving two
  # failing gates instead of one on the last round. The combined defect
  # score (clashes + 25 per extra component) accepts it: 117 -> 113.
  run15_prev = {"unit_ok": True, "ifc_ok": True, "frame_ok": True,
                "clashes": 92, "max_components": 2, "rows_below_80": 0,
                "components": {"tray": 1, "busway": 1, "chw": 2}}
  run15_cur = {"unit_ok": True, "ifc_ok": True, "frame_ok": True,
               "clashes": 113, "max_components": 1, "rows_below_80": 0,
               "components": {"tray": 1, "busway": 1, "chw": 1}}
  check("run-15 defect scores",
        serve._defects(run15_prev) == 117
        and serve._defects(run15_cur) == 113,
        (serve._defects(run15_prev), serve._defects(run15_cur)))
  check("run-15 CONN fix at +21 clashes is accepted, not reverted",
        not serve._is_regression(run15_prev, run15_cur))
  # Run 12: a fix cut clashes 92 -> 0 by shortening segments short of
  # their neighbours (tray 1 -> 7, CHW 4 -> 8). The defect score still
  # reverts it (167 -> 325), so the run-12 tightening survives the new
  # rule.
  run12_prev = {"unit_ok": True, "ifc_ok": True, "frame_ok": True,
                "clashes": 92, "max_components": 4, "rows_below_80": 0,
                "components": {"tray": 1, "busway": 1, "chw": 4}}
  run12_cur = {"unit_ok": True, "ifc_ok": True, "frame_ok": True,
               "clashes": 0, "max_components": 8, "rows_below_80": 0,
               "components": {"tray": 7, "busway": 1, "chw": 8}}
  check("run-12 defect scores",
        serve._defects(run12_prev) == 167
        and serve._defects(run12_cur) == 325,
        (serve._defects(run12_prev), serve._defects(run12_cur)))
  check("run-12 disconnect-for-clashes still reverts",
        serve._is_regression(run12_prev, run12_cur))
  # A fatal flip still reverts even when the defect score improves.
  check("fatal flip reverts despite a lower defect score",
        serve._is_regression(run15_prev,
                             {**run15_cur, "unit_ok": False}))
  check("inconclusive scores give no vector",
        serve._gate_score_vector({}, {}, {}, {}) is None)
  check("score vector built",
        serve._gate_score_vector(
            {"unit_ok": True, "scale": 1.0},
            {"ifc_ok": True, "violations": 0},
            {"clashes": 10, "frame_ok": True, "frame_outside": 0},
            {"components": {"tray": 5}, "rows_below_80": 2})
        == {"unit_ok": True, "ifc_ok": True, "frame_ok": True,
            "clashes": 10, "max_components": 5,
            "components": {"tray": 5}, "rows_below_80": 2})


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


async def test_run15_fix_accepted_then_reverted():
  print("run-15 oscillation: a CONN fix at +21 clashes is kept, "
        "a worse round after it is reverted, and the revert is noted")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"

  def sep_fail(n):
    return (
        f"exit code 0\nSEP collisions: {n}\n"
        f"SEP FAIL: {n} clashes (boxes overlapping > 2 mm on all axes)\n"
        "SEP_GATE: FAIL")

  def conn_fail(chw):
    return (
        "exit code 0\nCONN tray: 1 connected component(s)\n"
        "CONN busway: 1 connected component(s)\n"
        f"CONN chw: {chw} connected component(s)\n"
        f"CONN FAIL: chw: {chw} disconnected components\n"
        "CONN_GATE: FAIL")

  conn_pass = (
      "exit code 0\nCONN tray: 1 connected component(s)\n"
      "CONN busway: 1 connected component(s)\n"
      "CONN chw: 1 connected component(s)\n"
      "CONN_GATE: PASS")

  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS", "PASS"],
      },
      bash_outputs=[
          # Round 1: SEP 92, CONN FAIL (CHW 2 components).
          unit_pass, ifc_pass, sep_fail(92), conn_fail(2),
          # Round 2: the fix closed CONN (1 component) at +21 clashes
          # (SEP 113). Defects 117 -> 113: accepted, no restore, so no
          # extra gate outputs are consumed here.
          unit_pass, ifc_pass, sep_fail(113), conn_pass,
          # Round 3: SEP back to 92 but CONN broken again (2 components).
          # Defects 113 -> 117: a regression — the guard restores the
          # round-2 model and re-runs the gates (113 again, CONN PASS).
          unit_pass, ifc_pass, sep_fail(92), conn_fail(2),
          unit_pass, ifc_pass, sep_fail(113), conn_pass,
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
  check("round-2 fix accepted, not reverted",
        "113 clashes" in worker.resumes[1]
        and "made the model worse" not in worker.resumes[1],
        worker.resumes[1])
  check("round-2 gaps are the 113-clash model",
        "113 clashes" in worker.resumes[1]
        and "92 clashes" not in worker.resumes[1], worker.resumes[1])
  check("revert note appended to the final summary",
        "Note: a fix round made the model worse" in result, repr(result))
  check("revert note names both models",
        "92 clashes" in result and "113 clashes" in result, repr(result))
  check("revert note says the submitted model is the restored one",
        "The submitted model is the restored one" in result, repr(result))
  check("submitted model is the round-2 one (CONN PASS)",
        "Connectivity gate: no failures reported" in result, repr(result))
  check("gate scripts written to the task container",
        session.files.get("/tmp/sep_gate.py") == serve.SEP_GATE_SCRIPT.encode()
        and session.files.get("/tmp/conn_gate.py")
        == serve.CONN_GATE_SCRIPT.encode(),
        list(session.files))
  check("gate scripts logged", any("GATE_SCRIPTS" in log for log in session.logs),
        session.logs)


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
  check("ground truth prompt",
        "Drawing dimensions are in ground_truth.json" in prompt)
  check("no PNG decoding by the agent",
        "do not decode the PNGs yourself" in prompt)
  check("PNG reader deleted", "def read_png" not in prompt)
  check("west-strip trunk rule", "west strip" in prompt)
  check("band arithmetic in the tier rule",
        "742 mm" in prompt and "515 mm" in prompt)
  check("butt-joint rule", "butt joints" in prompt)
  check("profile orientation rule", "127 mm" in prompt)
  check("drops go down", "goes down to the CRAH top" in prompt)
  check("tier rule", "separate tiers" in prompt)
  verifier = serve.Agent(session, role="verifier")
  vprompt = verifier.system_prompt("")
  check("verifier connectivity checklist",
        "recompute the route geometry" in vprompt)
  check("verifier quotes numbers", "a PASS that does not quote them" in vprompt)
  check("verifier distrusts empty clash checks",
        "did not actually measure anything" in vprompt)
  check("verifier re-runs the gate scripts",
        "python3 /tmp/sep_gate.py" in vprompt
        and "python3 /tmp/conn_gate.py" in vprompt)
  check("verifier warned about geom metres",
        "returns coordinates in metres" in vprompt)
  check("verifier gets the IFC snippet",
        "ifcopenshell.validate.validate(model, _log, express_rules=True)"
        in vprompt)
  check("verifier gets the routing guide", "one header per row" in vprompt)
  session2 = FakeSession(21600)
  session2.instruction = "Design a steel frame."
  worker2 = serve.Agent(session2, role="worker")
  check("no routing guide without routing",
        "one header per row" not in worker2.system_prompt(""))
  check("no ground truth prompt without routing",
        "ground_truth.json" not in worker2.system_prompt(""))


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
          "below the sprinkler deflector plane" in out, out)
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


def test_sep_gate_rack_diagnostic():
  print("separation gate names the racks overlapping CRAHs")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import shutil
  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    # Run 15's defect, shrunk: a 1200x900 CRAH at (4528, 2996) with two
    # 600x1000 racks at (4200, 3000) and (4800, 3000) — both overlap it
    # — plus one rack far away that does not.
    outdir = os.path.join(tmp, "rack", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcFurniture', 'CRAH-A01', 1200, 900, 2200, 4528, 2996, 0),
        ('IfcFurniture', 'Rack-A-4', 600, 1000, 2000, 4200, 3000, 0, 'Rack'),
        ('IfcFurniture', 'Rack-A-5', 600, 1000, 2000, 4800, 3000, 0, 'Rack'),
        ('IfcFurniture', 'Rack-A-9', 600, 1000, 2000, 9000, 3000, 0, 'Rack'),
    ])
    out = _run_sep_gate(os.path.join(tmp, "rack"))
    check("rack x CRAH pairs counted", "SEP collisions: 2" in out, out)
    check("gate fails", "SEP_GATE: FAIL" in out, out)
    check("overlapping racks named",
          "2 racks overlap CRAHs or columns" in out, out)
    check("first overlapping rack named", "Rack-A-4" in out, out)
    check("second overlapping rack named", "Rack-A-5" in out, out)
    check("non-overlapping rack not named", "Rack-A-9" not in out, out)
    check("skip rule quoted",
          "within 900 mm in x and 950 mm in y" in out, out)
    # No racks at all: no diagnostic line, and the gate passes.
    outdir2 = os.path.join(tmp, "norack", "output")
    os.makedirs(outdir2)
    _build_test_ifc(os.path.join(outdir2, "model.ifc"), [
        ('IfcFurniture', 'CRAH-A01', 1200, 900, 2200, 4528, 2996, 0),
    ])
    out2 = _run_sep_gate(os.path.join(tmp, "norack"))
    check("no racks, no diagnostic",
          "racks overlap" not in out2, out2)
    check("no racks, gate passes", "SEP_GATE: PASS" in out2, out2)


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


def _synthetic_ground_truth():
  """A small ground truth (2 rows) for the gate and guard tests."""
  return {
      "valid": True,
      "hall": {"width_mm": 27000, "height_mm": 21000,
               "tiles_x": 45, "tiles_y": 35},
      "rows": [
          {"name": "A", "y_centre": 3000, "y_band": [2400, 3600],
           "x_start": 1800, "x_end": 19200},
          {"name": "B", "y_centre": 6000, "y_band": [5400, 6600],
           "x_start": 1800, "x_end": 19800},
      ],
      "crahs": [
          {"x": 4500, "y": 3000, "tag": "CRAH-A01", "type": "in-row",
           "dn": 50},
          {"x": 9500, "y": 6000, "tag": "CRAH-B01", "type": "in-row",
           "dn": 50},
      ],
      "entry_drawn": {"xy": [27, 9889], "tile": "A17"},
      "entry_grader": {"xy": [200, 3360],
                       "source": "prior grader output"},
      "entry_discrepancy_mm": 6532,
      "columns": [],
      "pdus": [],
      "sprinklers": {"ceiling": [], "in_aisle": []},
      "scale_mm_per_px": 38.04,
      "profiles": {"tray": {"width": 683, "height": 127},
                   "busway": {"width": 128, "height": 147},
                   "chw": {"od": 114.3, "od_insulated": 168.3}},
      "band": {"overhead": [2600, 3115], "underfloor": [-865, 0],
               "underfloor_top_clearance": 20,
               "datums": {"soffit": 3600, "deflector": 3574.6,
                          "clear_height": 2600}},
      "rules": {"tray_busway_separation": 300,
                "pdu_dedicated_space": 1800,
                "sprinkler_standoff": 460},
      "extraction_method": {},
      "errors": [],
  }


def test_draw_tools_geometry():
  print("draw_tools geometry extraction on the real drawings")
  try:
    import numpy  # noqa: F401
  except ImportError:
    print("  skipped: numpy not installed")
    return
  import draw_tools
  harness_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  input_dir = os.path.join(
      harness_dir, "..", "tasks", "data-hall-mep-routing", "environment",
      "filesystem", "dc_model__hall_seed000_d000__mep_routing", "input")
  if not os.path.isdir(input_dir):
    print("  skipped: smoke-task drawings not present")
    return
  gt, errors, warnings = draw_tools.extract(input_dir)
  check("extraction succeeded", gt is not None, errors)
  check("no errors or warnings", errors == [] and warnings == [],
        (errors, warnings))
  check("hall size",
        gt["hall"]["width_mm"] == 27000
        and gt["hall"]["height_mm"] == 21000, gt["hall"])
  check("scale ~38.03 mm/px",
        abs(gt["scale_mm_per_px"] - 38.03) < 0.1,
        gt["scale_mm_per_px"])
  # Row centrelines from the A-601 tile schedule (the authoritative
  # values): geometry must land within 300 mm of them.
  expected = {"A": 3000, "B": 6000, "C": 8400,
              "D": 11400, "E": 13800, "F": 16800}
  check("6 rows", len(gt["rows"]) == 6, gt["rows"])
  for row in gt["rows"]:
    check(f"row {row['name']} centreline within 300 mm",
          abs(row["y_centre"] - expected[row["name"]]) <= 300,
          row["y_centre"])
  check("22 CRAHs", len(gt["crahs"]) == 22, len(gt["crahs"]))
  check("4 perimeter + 18 in-row CRAHs",
        sum(1 for c in gt["crahs"] if c["type"] == "perimeter") == 4
        and sum(1 for c in gt["crahs"] if c["type"] == "in-row") == 18)
  entry = gt["entry_drawn"]["xy"]
  check("entry on the west wall", entry[0] < 1000, entry)
  check("entry near the validated (46, 9887)",
        abs(entry[0] - 46) <= 300 and abs(entry[1] - 9887) <= 300, entry)
  check("entry tile A17", gt["entry_drawn"]["tile"] == "A17",
        gt["entry_drawn"])
  check("entry discrepancy recorded",
        abs(gt["entry_discrepancy_mm"] - 6532) < 100,
        gt["entry_discrepancy_mm"])
  check("6 columns", len(gt["columns"]) == 6, gt["columns"])
  check("6 PDUs", len(gt["pdus"]) == 6, gt["pdus"])
  check("63 ceiling + 16 in-aisle sprinklers",
        len(gt["sprinklers"]["ceiling"]) == 63
        and len(gt["sprinklers"]["in_aisle"]) == 16,
        (len(gt["sprinklers"]["ceiling"]),
         len(gt["sprinklers"]["in_aisle"])))


def test_vision_parse():
  print("vision reply parsing and ground-truth merge")
  reply = """{
    "rows": [
      {"name": "A", "first_tile": "D05", "last_tile": "AF05"},
      {"name": "B", "first_tile": "D10", "last_tile": "AG10"}
    ],
    "profiles": {"tray": {"width": 683, "height": 127},
                 "busway": {"width": 128, "height": 147},
                 "chw": {"od": 114.3, "od_insulated": 168.3}},
    "band": {"overhead": [2600, 3115], "underfloor": [-865, 0],
             "underfloor_top_clearance": 20},
    "rules": {"tray_busway_separation": 300,
              "pdu_dedicated_space": 1800,
              "sprinkler_standoff": 460},
    "datums": {"soffit": 3600, "deflector": 3574.6,
               "clear_height": 2600}
  }"""
  vision = serve._parse_vision_reply(reply)
  check("rows parsed", len(vision.get("rows", [])) == 2, vision)
  check("row A tile math",
        vision["rows"][0] == {"name": "A", "y_centre": 3000,
                              "y_band": [2400, 3600],
                              "x_start": 1800, "x_end": 19200},
        vision["rows"][0])
  check("row B tile math",
        vision["rows"][1]["y_centre"] == 6000
        and vision["rows"][1]["x_end"] == 19800, vision["rows"][1])
  check("profiles parsed",
        vision["profiles"]["tray"] == {"width": 683, "height": 127},
        vision.get("profiles"))
  check("band parsed", vision["band"]["overhead"] == [2600, 3115],
        vision.get("band"))
  check("rules parsed",
        vision["rules"]["tray_busway_separation"] == 300,
        vision.get("rules"))
  check("datums parsed", vision["datums"]["deflector"] == 3574.6,
        vision.get("datums"))
  check("tile column AF is 32", serve._tile_column("AF") == 32)
  check("tile D05 to mm",
        serve._tile_row_mm("D05") == (1800, 3000, 3600),
        serve._tile_row_mm("D05"))
  # Garbage and fenced JSON.
  check("garbage reply parses to empty",
        serve._parse_vision_reply("no json here") == {})
  check("fenced JSON parses",
        serve._parse_vision_reply("```json\n" + reply + "\n```") == vision)
  # Merge: tile values replace geometry ones; a >500 mm disagreement
  # warns and the tile schedule still wins.
  gt = _synthetic_ground_truth()
  gt["rows"][0]["y_centre"] = 2986  # geometry value, 14 mm off
  gt["rows"][1]["y_centre"] = 5200  # geometry value, 800 mm off
  warnings = serve._merge_vision_ground_truth(gt, vision)
  check("tile schedule wins for row A",
        gt["rows"][0]["y_centre"] == 3000, gt["rows"][0])
  check("tile schedule wins for row B despite the gap",
        gt["rows"][1]["y_centre"] == 6000, gt["rows"][1])
  check("disagreement warned",
        len(warnings) == 1 and "B" in warnings[0], warnings)
  check("profiles merged", gt["profiles"]["chw"]["od_insulated"] == 168.3)
  check("band merged with datums",
        gt["band"]["datums"]["deflector"] == 3574.6, gt["band"])


async def test_empty_submission_guard():
  print("empty-submission guard writes a fallback model")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  session.files["output/solution.py"] = (
      b"# submission stub was never replaced\n")
  FakeAgent.reset(scripts={"planner": ["p"], "worker": [""]})
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("guard logged loudly",
        any("EMPTY SUBMISSION" in t for t in session.logs), session.logs)
  check("failure report returned, not an empty string",
        "failed to produce a model" in result, repr(result))
  check("final message is the failure report",
        session.final_message == result, repr(session.final_message))
  fallback = session.files["output/solution.py"].decode()
  check("fallback written, not the stub",
        "submission stub was never replaced" not in fallback, fallback[:200])
  check("fallback is the harness model",
        "Last-resort model" in fallback, fallback[:200])
  check("fallback declares millimetres",
        "MILLI" in fallback and "IfcUnitAssignment" in fallback)


async def test_planner_skipped_with_ground_truth():
  print("planner skipped when ground truth exists")
  import json
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  session.files["ground_truth.json"] = json.dumps(
      _synthetic_ground_truth()).encode()
  FakeAgent.reset(scripts={"worker": ["worker summary"],
                           "verifier": ["PASS"]})
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("no planner phase", FakeAgent.roles() == ["worker", "verifier"],
        FakeAgent.roles())
  check("worker got the ground-truth summary",
        "Ground truth measured from the drawings"
        in FakeAgent.first("worker").plan,
        FakeAgent.first("worker").plan)
  check("merged ground truth copied to output/",
        "output/ground_truth.json" in session.files
        and json.loads(session.files["output/ground_truth.json"])["rows"][0]
        ["name"] == "A",
        list(session.files))
  check("summary names the rows",
        "A 3000" in FakeAgent.first("worker").plan
        and "B 6000" in FakeAgent.first("worker").plan)
  check("result verified",
        result.endswith("Independent verification: PASS."), repr(result))


def test_conn_gate_ground_truth():
  print("connectivity gate measures against ground_truth.json")
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

  # A route that is perfect against its own routing.json (entry at
  # (200, 3360), one row at y=3360) but nowhere near the ground truth
  # (entry (27, 9889), rows at y=3000/6000 spanning x 1800-19200).
  segments = [
      ('IfcCableCarrierSegment', 'CableTray-1', 1800, 600, 300, 1100, 3360, 2600),
      ('IfcCableCarrierSegment', 'CableTray-2', 600, 1640, 300, 2000, 4180, 2600),
      ('IfcCableCarrierSegment', 'Busway-1', 1800, 600, 300, 1100, 3960, 2600),
      ('IfcCableCarrierSegment', 'Busway-2', 600, 1640, 300, 2000, 4780, 2600),
      ('IfcPipeSegment', 'CHW-1', 1800, 600, 300, 1100, 2760, 2600),
      ('IfcPipeSegment', 'CHW-2', 600, 2240, 300, 2000, 3880, 2600),
      ('IfcPipeSegment', 'CHW-Drop-1', 300, 300, 1600, 2000, 5000, 1000),
  ]
  routing = {
      'entry': [200, 3360, 0],
      'rows': [{'name': 'A', 'start': [1100, 3360, 0],
                'end': [2000, 3360, 0]}],
      'crahs': [[2000, 5000, 0]],
  }
  with tempfile.TemporaryDirectory() as tmp:
    outdir = os.path.join(tmp, "gt", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), segments)
    json.dump(routing, open(os.path.join(outdir, "routing.json"), "w"))
    json.dump(_synthetic_ground_truth(),
              open(os.path.join(tmp, "gt", "ground_truth.json"), "w"))
    out = run_gate(os.path.join(tmp, "gt"))
    check("gate measures against ground truth",
          "CONN measuring against ground_truth.json" in out, out)
    check("self-consistent model fails against ground truth",
          "CONN_GATE: FAIL" in out, out)
    check("entry distance measured from the drawn entry",
          "nothing within 2500 mm of the entry" in out, out)
    check("row coverage measured from the true rows",
          "row A: tray covers only 16%" in out, out)

    # Invalid ground truth falls back to routing.json — and the same
    # model passes against its own declaration.
    outdir = os.path.join(tmp, "fallback", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), segments)
    json.dump(routing, open(os.path.join(outdir, "routing.json"), "w"))
    json.dump({"valid": False},
              open(os.path.join(tmp, "fallback", "ground_truth.json"), "w"))
    out = run_gate(os.path.join(tmp, "fallback"))
    check("invalid ground truth falls back to routing.json",
          "output/routing.json (fallback: no ground truth)" in out, out)
    check("same model passes against routing.json",
          "CONN_GATE: PASS" in out, out)


def test_sep_gate_ground_truth():
  print("separation gate reads the band and deflector from ground truth")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import json
  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    # One tray segment topping at 3300 mm, no sprinkler entities in the
    # model: without ground truth the sprinkler check is skipped and the
    # gate passes; with it the deflector plane (3574.6) catches the
    # segment — 3300 is only 275 mm below the plane, not 460.
    outdir = os.path.join(tmp, "high", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 10000, 683, 150,
         5000, 5000, 3150),
    ])
    out = _run_sep_gate(os.path.join(tmp, "high"))
    check("no sprinklers in the model: check skipped, gate passes",
          "SEP_GATE: PASS" in out, out)
    json.dump(_synthetic_ground_truth(),
              open(os.path.join(tmp, "high", "ground_truth.json"), "w"))
    out = _run_sep_gate(os.path.join(tmp, "high"))
    check("deflector plane from ground truth catches the high segment",
          "below the sprinkler deflector plane" in out, out)
    check("gate fails with ground truth", "SEP_GATE: FAIL" in out, out)

    # Two services sharing a band: the tier message now names the band
    # height from ground truth instead of the old placeholder tiers.
    outdir = os.path.join(tmp, "flat", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-1', 10000, 600, 150,
         5000, 5000, 2850),
        ('IfcCableCarrierSegment', 'Busway-1', 10000, 600, 150,
         5000, 5000, 2850),
    ])
    json.dump(_synthetic_ground_truth(),
              open(os.path.join(tmp, "flat", "ground_truth.json"), "w"))
    out = _run_sep_gate(os.path.join(tmp, "flat"))
    check("tier message names the band height",
          "overhead band is only 515 mm high" in out, out)
    check("no placeholder tiers in the message",
          "tray z=3400" not in out, out)


def test_fallback_solution():
  print("fallback solution runs and passes the gates")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import json
  import subprocess
  import tempfile

  def run_script(script_text, workdir, name):
    script = os.path.join(workdir, name)
    with open(script, "w") as f:
      f.write(script_text)
    r = subprocess.run([sys.executable, script], cwd=workdir,
                       capture_output=True, text=True)
    return r.stdout + r.stderr

  with tempfile.TemporaryDirectory() as tmp:
    gt = _synthetic_ground_truth()
    source = serve._fallback_solution(gt)
    check("fallback source is not the stub",
          "submission stub was never replaced" not in source)
    check("fallback carries the unit block",
          "IfcUnitAssignment" in source and "MILLI" in source)
    os.makedirs(os.path.join(tmp, "output"))
    json.dump(gt, open(os.path.join(tmp, "ground_truth.json"), "w"))
    with open(os.path.join(tmp, "output", "solution.py"), "w") as f:
      f.write(source)
    r = subprocess.run([sys.executable, "output/solution.py"], cwd=tmp,
                       capture_output=True, text=True)
    check("fallback solution runs", r.returncode == 0, r.stdout + r.stderr)
    check("fallback writes model.ifc",
          os.path.exists(os.path.join(tmp, "output", "model.ifc")))
    out = run_script(serve.UNIT_GATE_SCRIPT, tmp, "unit_gate.py")
    check("fallback units pass", "UNIT_GATE: PASS" in out, out)
    try:
      import pytest  # noqa: F401
    except ImportError:
      # ifcopenshell's express-rules engine imports _pytest; the task
      # container has pytest, this env may not. Run the suite with
      # --with pytest to exercise this check.
      print("  skipped: IFC express validation needs pytest")
    else:
      out = run_script(serve.IFC_GATE_SCRIPT, tmp, "ifc_gate.py")
      check("fallback IFC validates", "IFC_GATE: PASS" in out, out)
    out = _run_sep_gate(tmp)
    check("fallback has no clashes", "SEP collisions: 0" in out, out)
    check("fallback separation passes", "SEP_GATE: PASS" in out, out)
    out = run_script(serve.CONN_GATE_SCRIPT, tmp, "conn_gate.py")
    check("fallback connectivity passes", "CONN_GATE: PASS" in out, out)

    # No ground truth at all: still a runnable model, not a stub.
    source = serve._fallback_solution(None)
    with open(os.path.join(tmp, "output", "solution.py"), "w") as f:
      f.write(source)
    r = subprocess.run([sys.executable, "output/solution.py"], cwd=tmp,
                       capture_output=True, text=True)
    check("no-ground-truth fallback runs", r.returncode == 0,
          r.stdout + r.stderr)


async def test_standalone_check_blocks_submission():
  print("standalone check catches a script that opens ground_truth.json")
  # The run-12 defect: solution.py reads ground_truth.json at runtime and
  # dies in the grader's empty directory (0.0 in 0.4 s). The gates pass and
  # the verifier passes, but the standalone check fails every round; the
  # fix rounds cannot fix it (the canned worker never rewrites the
  # script), so with the rounds spent the harness replaces it with the
  # fallback instead of submitting a script that cannot run.
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  session.files["output/solution.py"] = (
      b"import json\n"
      b"gt = json.load(open('ground_truth.json'))\n"
      b"print('built from', gt['rows'][0]['name'])\n")
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"
  sep_pass = "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS"
  conn_pass = "exit code 0\nCONN_GATE: PASS"
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS", "PASS"],
      },
      bash_outputs=[unit_pass, ifc_pass, sep_pass, conn_pass] * 4,
      standalone_result="STANDALONE_EXIT:1\nSTANDALONE_IFC:MISSING",
      standalone_tail=(
          "FileNotFoundError: [Errno 2] No such file or directory:"
          " 'ground_truth.json'"),
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  worker = FakeAgent.first("worker")
  check("two fix rounds used", len(worker.resumes) == 2, worker.resumes)
  check("standalone gap reached the worker",
        any("STANDALONE FAIL" in r for r in worker.resumes),
        worker.resumes)
  check("standalone gap names the fix",
        any("Copy every number" in r for r in worker.resumes),
        worker.resumes)
  check("fallback report returned, not a pass",
        "did not run alone" in result, repr(result))
  check("not marked verified",
        "Independent verification: PASS." not in result, repr(result))
  check("fallback replaced the script",
        "Last-resort model" in session.files["output/solution.py"].decode(),
        session.files["output/solution.py"][:200])


async def test_standalone_check_passes_clean():
  print("standalone check does not block a self-contained script")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"
  sep_pass = "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS"
  conn_pass = "exit code 0\nCONN_GATE: PASS"
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS"],
      },
      bash_outputs=[unit_pass, ifc_pass, sep_pass, conn_pass],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("result verified",
        result.endswith("Independent verification: PASS."), repr(result))
  check("no standalone failure in the summary",
        "STANDALONE FAIL" not in result, repr(result))


async def test_empty_summary_still_verifies():
  print("empty worker summary still runs the gates (run 13)")
  # Run 13: the work phase died on a 503 ladder after solution.py was
  # written, so the worker produced no summary text — and solve()
  # returned "" without ever running the gates or the standalone check.
  # With a real solution.py the harness must fall through to the verify
  # loop instead: the gates run, the standalone failure reaches the
  # worker as a fix round, and the result is a non-empty report.
  import json
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  session.files["ground_truth.json"] = json.dumps(
      _synthetic_ground_truth()).encode()
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"
  sep_pass = "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS"
  conn_pass = "exit code 0\nCONN_GATE: PASS"
  FakeAgent.reset(
      scripts={"worker": [""], "verifier": ["PASS", "PASS", "PASS"]},
      bash_outputs=[unit_pass, ifc_pass, sep_pass, conn_pass] * 4,
      standalone_result="STANDALONE_EXIT:1\nSTANDALONE_IFC:MISSING",
      standalone_tail="NameError: name 'json' is not defined",
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("verify loop ran (verifier was created)",
        "verifier" in FakeAgent.roles(), FakeAgent.roles())
  check("verifier got the harness placeholder, not an empty context",
        "no summary text" in FakeAgent.first("verifier").context,
        FakeAgent.first("verifier").context[:200])
  worker = FakeAgent.first("worker")
  check("two fix rounds used", len(worker.resumes) == 2, worker.resumes)
  check("standalone gap reached the worker",
        any("STANDALONE FAIL" in r for r in worker.resumes),
        worker.resumes)
  check("result is not empty", bool(result), repr(result))
  check("fallback report returned, not a pass",
        "did not run alone" in result, repr(result))
  check("not marked verified",
        "Independent verification: PASS." not in result, repr(result))
  check("mandatory verification was logged",
        any("mandatory first verification" in t for t in session.logs),
        session.logs)


async def test_empty_summary_passes_clean():
  print("empty worker summary passes a clean model")
  # Same fall-through as above, but the gates and the standalone check
  # pass: the harness placeholder plus the verifier's PASS is a real
  # verification, and the result carries it.
  import json
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  session.files["ground_truth.json"] = json.dumps(
      _synthetic_ground_truth()).encode()
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"
  sep_pass = "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS"
  conn_pass = "exit code 0\nCONN_GATE: PASS"
  FakeAgent.reset(
      scripts={"worker": [""], "verifier": ["PASS"]},
      bash_outputs=[unit_pass, ifc_pass, sep_pass, conn_pass],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("result verified",
        result.endswith("Independent verification: PASS."), repr(result))
  check("result is the harness summary, not empty",
        "no summary text" in result, repr(result))
  check("no fix round needed", len(FakeAgent.first("worker").resumes) == 0,
        FakeAgent.first("worker").resumes)


async def test_empty_summary_verifier_fails_too():
  print("empty worker summary plus a failed verifier still returns text")
  # The endpoint is down for both phases: the worker wrote no summary
  # and the verifier returns nothing. The gates and the standalone check
  # pass, so there is nothing to fix — but the task must still end with
  # a non-empty harness summary, never a 0-byte final_message.
  import json
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water."
  session.files["ground_truth.json"] = json.dumps(
      _synthetic_ground_truth()).encode()
  unit_pass = (
      "exit code 0\nUNIT output/model.ifc: scale x1.000,"
      " millimetre unit declared: True\n"
      "UNIT_GATE: PASS (1 file(s), scale x1.0, millimetre units)")
  ifc_pass = "exit code 0\nIFC_GATE: PASS (1 file(s), 0 EXPRESS violations)"
  sep_pass = "exit code 0\nSEP collisions: 0\nSEP_GATE: PASS"
  conn_pass = "exit code 0\nCONN_GATE: PASS"
  FakeAgent.reset(
      scripts={"worker": [""], "verifier": [""]},
      bash_outputs=[unit_pass, ifc_pass, sep_pass, conn_pass],
  )
  real = serve.Agent
  serve.Agent = FakeAgent
  try:
    result = await serve.solve(session)
  finally:
    serve.Agent = real
  check("result is not empty", bool(result), repr(result))
  check("result is the harness summary",
        "no summary text" in result, repr(result))
  check("not marked verified",
        "Independent verification: PASS." not in result, repr(result))
  check("verifier failure was logged",
        any("verify phase returned no text" in t for t in session.logs),
        session.logs)


def test_correct_summary():
  print("final-message correction quotes the real gate numbers")
  verdict = ("Unit gate: no failures reported\n"
             "IFC gate: no failures reported\n"
             "Separation gate: FAIL — 0 collisions but a drop hangs\n"
             "Connectivity gate: FAIL — chw: 4 disconnected components")
  # Run 12's summary: PASS claims about the failing gates.
  summary = ("Clash count: 0 clashes (PASS)\n"
             "Separation: PASS\n"
             "Coverage: PASS\n"
             "Entry distance: PASS\n"
             "\nDeterministic gates:\n" + verdict)
  corrected = serve._correct_summary(summary, verdict)
  check("correction appended", "\n\nCORRECTION:" in corrected, corrected)
  check("contradicted clash claim quoted",
        "Clash count: 0 clashes (PASS)" in corrected, corrected)
  check("contradicted coverage claim quoted",
        "Coverage: PASS" in corrected, corrected)
  check("failing gate lines quoted",
        "Separation gate: FAIL" in corrected
        and "Connectivity gate: FAIL" in corrected, corrected)
  # An honest summary (PASS claims only about passing gates, the
  # failures named as failures) is left alone.
  honest = ("Unit gate PASS, IFC gate PASS, but the separation gate still"
            " fails and CHW has 4 components.\n\nDeterministic gates:\n"
            + verdict)
  check("honest summary unchanged",
        serve._correct_summary(honest, verdict) == honest)
  # A clean verdict needs no correction even with PASS claims.
  clean = ("Unit gate: no failures reported\n"
           "IFC gate: no failures reported\n"
           "Separation gate: no failures reported\n"
           "Connectivity gate: no failures reported")
  check("clean verdict unchanged",
        serve._correct_summary("All gates PASS", clean) == "All gates PASS")
  # A summary with no PASS claims is left alone (the verdict is already
  # appended below it).
  plain = ("The model still fails the separation gate.\n\n"
           "Deterministic gates:\n" + verdict)
  check("no PASS claim, no correction",
        serve._correct_summary(plain, verdict) == plain)
  # Run 15's summary: no "pass" anywhere, but two success claims about
  # the failing gates ("Resolved previous hard clashes", "all 34 CHW
  # segments form a continuous network"). The old pass-only detector
  # missed both; the phrasing detector must catch them and quote the
  # real gate numbers even though the Deterministic gates section
  # already lists them.
  run15 = ("Separation / Clash Gate: Resolved previous hard clashes by"
           " vertically and horizontally isolating services.\n"
           "Connectivity Gate: Resolved chilled water continuity gaps by"
           " adding the south perimeter CHW header (y = 469 mm) so that"
           " all 34 CHW segments form a continuous network reaching all"
           " 22 CRAH units within tolerance.\n"
           "\nDeterministic gates:\n" + verdict)
  corrected15 = serve._correct_summary(run15, verdict)
  check("run-15 correction appended",
        "\n\nCORRECTION:" in corrected15, corrected15)
  check("run-15 resolved-clashes claim quoted",
        "Resolved previous hard clashes" in corrected15, corrected15)
  check("run-15 continuous-network claim quoted",
        "form a continuous network" in corrected15, corrected15)
  check("run-15 failing gate lines quoted",
        "Separation gate: FAIL" in corrected15
        and "Connectivity gate: FAIL" in corrected15, corrected15)
  # The self-check phrasing "Zero hard clashes" is a success claim too.
  zero = ("Zero hard clashes: verified via separate vertical / horizontal"
          " zoning.\n\nDeterministic gates:\n" + verdict)
  check("zero-clashes claim corrected",
        "\n\nCORRECTION:" in serve._correct_summary(zero, verdict)
        and "Zero hard clashes" in serve._correct_summary(zero, verdict),
        serve._correct_summary(zero, verdict))
  # True claims about the same gates are NOT corrected: the standalone
  # run really did succeed, the per-row coverage really is 100% (the
  # gate agrees), and the entry distances are measurements, not success
  # claims. None of these lines contains a resolution word for its gate.
  true_claims = ("Re-ran output/solution.py successfully in a clean"
                 " standalone directory (exited 0, total elements: 265).\n"
                 "Schema Validation: Re-run using"
                 " ifcopenshell.validate.validate returned 0 errors.\n"
                 "Per-Row Coverage: Rows A, B, C, D, E, and F are served"
                 " along 100% of their length.\n"
                 "Entry Distance: CHW 486.50 mm, Cable Tray 286.50 mm.\n"
                 "\nDeterministic gates:\n" + verdict)
  check("true claims left alone",
        serve._correct_summary(true_claims, verdict) == true_claims,
        serve._correct_summary(true_claims, verdict))


def test_band_validation():
  print("vision band cross-checked against datums and rules")
  # Run 12's vision read: overhead [3085, 3600] (the top 515 mm below
  # the soffit), while the datums+rules give [2600, 3115].
  gt = {"rows": [], "band": {"overhead": [2600, 3115],
                             "datums": {"soffit": 3600, "deflector": 3574.6,
                                        "clear_height": 2600}}}
  vision = {"band": {"overhead": [3085, 3600]},
            "datums": {"soffit": 3600, "deflector": 3574.6,
                       "clear_height": 2600},
            "rules": {"sprinkler_standoff": 460}}
  warnings = serve._merge_vision_ground_truth(gt, vision)
  check("wrong band corrected to datums+rules",
        abs(gt["band"]["overhead"][0] - 2600) < 1
        and abs(gt["band"]["overhead"][1] - 3114.6) < 1, gt["band"])
  check("warning names the disagreement",
        any("datums+rules win" in w for w in warnings), warnings)
  # A band that already agrees is left alone.
  gt2 = {"rows": [], "band": {"overhead": [2600, 3115],
                              "datums": {"soffit": 3600, "deflector": 3574.6,
                                         "clear_height": 2600}}}
  vision2 = {"band": {"overhead": [2600, 3115]},
             "datums": {"soffit": 3600, "deflector": 3574.6,
                        "clear_height": 2600},
             "rules": {"sprinkler_standoff": 460}}
  warnings2 = serve._merge_vision_ground_truth(gt2, vision2)
  check("correct band untouched",
        gt2["band"]["overhead"] == [2600, 3115], gt2["band"])
  check("no warning for a correct band", warnings2 == [], warnings2)


def test_sep_gate_riser():
  print("separation gate accepts a plenum riser, rejects a hanging drop")
  try:
    import ifcopenshell  # noqa: F401
  except ImportError:
    print("  skipped: ifcopenshell not installed")
    return
  import tempfile

  with tempfile.TemporaryDirectory() as tmp:
    # CHW in the under-floor plenum: the header runs at z -500..-332 and
    # the riser stands on it (bottom z=-332, top z=0). Under the old
    # top-only rule this failed — the top is 416 mm from the header's
    # centreline — but the bottom touches the header, so it is connected.
    outdir = os.path.join(tmp, "riser", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcPipeSegment', 'CHW-SouthPerimeter', 10000, 168, 168,
         5000, 5000, -500),
        ('IfcPipeSegment', 'CHW-Riser-1', 168, 168, 332,
         5000, 5000, -332),
    ])
    out = _run_sep_gate(os.path.join(tmp, "riser"))
    check("plenum riser passes", "SEP_GATE: PASS" in out, out)
    check("riser is not called a failed drop",
          "does not hang from the header" not in out, out)

    # A truly hanging drop: the CHW header runs at z 2600..2768 along
    # y=5000 and the vertical segment at (5000, 9000) spans z 1000..2000
    # — neither end is near a horizontal segment. The message must name
    # the nearer end (the top) and the direction (south).
    outdir = os.path.join(tmp, "hang", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcPipeSegment', 'CHW-Header', 10000, 168, 168,
         5000, 5000, 2600),
        ('IfcPipeSegment', 'CHW-Drop-1', 168, 168, 1000,
         5000, 9000, 1000),
    ])
    out = _run_sep_gate(os.path.join(tmp, "hang"))
    check("hanging drop fails", "SEP_GATE: FAIL" in out, out)
    check("hanging drop message names the top",
          "does not hang from the header" in out, out)
    check("hanging drop message gives the direction",
          "to the south" in out, out)

    # A drop built upward from an overhead header: its bottom touches the
    # header, but that is not a riser — a riser stands on an under-floor
    # header. The run-7 model did exactly this and the old top-only rule
    # caught it; the both-ends rule must still catch it.
    outdir = os.path.join(tmp, "upward", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcPipeSegment', 'CHW-Header', 10000, 168, 168,
         5000, 5000, 2600),
        ('IfcPipeSegment', 'CHW-Drop-1', 168, 168, 1232,
         5000, 5000, 2768),
    ])
    out = _run_sep_gate(os.path.join(tmp, "upward"))
    check("upward drop fails", "SEP_GATE: FAIL" in out, out)
    check("upward drop is not passed as a riser",
          "does not hang from the header" in out, out)
    check("upward drop message says it goes the wrong way",
          "goes up instead of down" in out, out)


def test_conn_gate_entry_aabb():
  print("connectivity gate measures entry distance to the box footprint")
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
    # The entry (27, 9889) is inside the feeder's footprint (x -973..1027,
    # y 9308..9991) but 239 mm from its longest-axis centreline (y=9650):
    # the old centreline rule reported 239 mm for a feeder whose box
    # contains the entry point; the footprint rule reports 0.
    outdir = os.path.join(tmp, "entry", "output")
    os.makedirs(outdir)
    _build_test_ifc(os.path.join(outdir, "model.ifc"), [
        ('IfcCableCarrierSegment', 'CableTray-EntryFeeder', 2000, 683, 127,
         27, 9650, 2600),
    ])
    json.dump(_synthetic_ground_truth(),
              open(os.path.join(tmp, "entry", "ground_truth.json"), "w"))
    out = run_gate(os.path.join(tmp, "entry"))
    check("entry inside the footprint measures 0 mm",
          "CONN tray: entry distance 0 mm" in out, out)


async def test_retry_429_capped():
  print("429 retries capped at 3 per call, 503 keeps full waits")
  import openai
  from types import SimpleNamespace
  from unittest import mock

  def status_error(status):
    # APIStatusError needs a response-like object with .request,
    # .status_code and .headers.get — no real HTTP response required.
    resp = SimpleNamespace(status_code=status, request=SimpleNamespace(),
                           headers={})
    return openai.APIStatusError("endpoint error", response=resp, body=None)

  async def run_with_errors(agent, errors):
    calls = []
    waits = []

    async def failing_stream(*args, **kwargs):
      calls.append(1)
      raise errors[len(calls) - 1]

    agent.stream = failing_stream

    async def fake_sleep(seconds):
      waits.append(seconds)

    with mock.patch("asyncio.sleep", new=fake_sleep):
      try:
        await agent.complete()
        return calls, waits, None
      except openai.APIStatusError as exc:
        return calls, waits, exc

  session = FakeSession(21600)
  agent = serve.Agent(session, role="worker")
  calls, waits, exc = await run_with_errors(
      agent, [status_error(429)] * 10)
  check("429 raises after the cap", exc is not None and exc.status_code == 429)
  check("429 attempts: 1 + 3 retries", len(calls) == 4, calls)
  check("429 waits are 5/15/30", waits == [5, 15, 30], waits)

  # 503 (server overload, not quota) keeps the full retry schedule.
  agent2 = serve.Agent(session, role="worker")
  calls, waits, exc = await run_with_errors(
      agent2, [status_error(503)] * 10)
  check("503 raises after the full schedule",
        exc is not None and exc.status_code == 503)
  check("503 attempts: 1 + 5 retries", len(calls) == 6, calls)
  check("503 waits are the full schedule",
        waits == [5, 15, 30, 60, 120], waits)

  # The global budget: 10 consecutive 429s already seen means the quota
  # is gone for this phase — the next 429 is not retried at all.
  agent3 = serve.Agent(session, role="worker")
  agent3._consec_429 = 10
  calls, waits, exc = await run_with_errors(
      agent3, [status_error(429)] * 10)
  check("budget exhausted: no retry",
        len(calls) == 1 and waits == [], (calls, waits))
  check("budget exhausted: 429 raised",
        exc is not None and exc.status_code == 429)

  # A successful call resets the budget.
  agent4 = serve.Agent(session, role="worker")
  agent4._consec_429 = 9
  agent4.record = lambda reply: asyncio.sleep(0)  # no-op record
  agent4.stream = lambda *a, **k: asyncio.sleep(
      0, result=SimpleNamespace(finish="stop"))
  reply = await agent4.complete()
  check("success returns the reply", reply.finish == "stop")
  check("success resets the 429 budget", agent4._consec_429 == 0)


def test_routing_fix_guidance():
  print("routing guide: self-containment and fix-round rules")
  session = FakeSession(21600)
  session.instruction = "Route the cable tray, busway and chilled water in the data hall."
  worker = serve.Agent(session, role="worker")
  prompt = worker.system_prompt("")
  check("self-containment: verifier runs solution.py alone",
        "copies ONLY output/solution.py" in prompt)
  check("self-containment: no runtime open()",
        "NEVER call open() on ground_truth.json" in prompt)
  check("self-containment: numbers as literals",
        "as literals" in prompt)
  check("fix must never break a connected joint",
        "must never break a joint that was connected" in prompt)
  check("bridge the named break exactly",
        "bridge exactly that gap" in prompt)
  check("riser is not a drop", "A riser is not a drop" in prompt)
  check("do not shift a header off its row",
        "Do not shift a header off its row" in prompt)
  check("sprinkler bases from the numbers",
        "Set tier heights from the sprinkler numbers" in prompt)
  check("drop direction from the gate",
        "the gate measured it in 3D" in prompt)
  check("import every module",
        "Import every module solution.py uses" in prompt)
  check("fix the crash and re-run",
        "do not stop while the last run crashed" in prompt)
  check("CRAH replaces racks",
        "CRAH replaces racks" in prompt)
  check("trunk must touch every header",
        "A trunk must touch every header it feeds" in prompt)
  check("evidence: fix a failed check and re-run",
        "never finish while the last run of a deliverable crashed"
        in prompt)
  # Run 15: the agent's own clash script reported 0 while the gate
  # measured 92 (ifcopenshell.geom returns metres); the guide now names
  # the harness gate scripts as the only clash measurement, with the
  # exact rack-skip predicate and the CHW drop recipe.
  check("gate scripts are the clash check",
        "python3 /tmp/sep_gate.py" in prompt
        and "python3 /tmp/conn_gate.py" in prompt)
  check("geom returns metres, not millimetres",
        "returns coordinates in METRES" in prompt)
  check("rack skip predicate with exact numbers",
        "abs(cx_rack - cx_crah) < 900" in prompt
        and "abs(cy_rack - cy_crah) < 950" in prompt)
  check("CHW drop recipe",
        "CHW drop recipe" in prompt
        and "header_z + 168.3" in prompt)
  check("column jog is at least 100 mm",
        "jog of at least 100 mm" in prompt)
  check("bridge ends at the other's face",
        "End the extended segment at the other's FACE" in prompt)


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
  test_sep_gate_rack_diagnostic()
  test_sep_gate_z_band_and_drop()
  test_merge_gate_gaps()
  await test_gates_run_in_parallel_and_merge()
  await test_gate_verdict_appended_when_rounds_run_out()
  await test_regression_guard_restores_worse_fix()
  await test_run15_fix_accepted_then_reverted()
  test_draw_tools_geometry()
  test_vision_parse()
  await test_empty_submission_guard()
  await test_planner_skipped_with_ground_truth()
  test_conn_gate_ground_truth()
  test_sep_gate_ground_truth()
  test_fallback_solution()
  await test_standalone_check_blocks_submission()
  await test_standalone_check_passes_clean()
  await test_empty_summary_still_verifies()
  await test_empty_summary_passes_clean()
  await test_empty_summary_verifier_fails_too()
  test_correct_summary()
  test_band_validation()
  test_sep_gate_riser()
  test_conn_gate_entry_aabb()
  await test_retry_429_capped()
  test_routing_fix_guidance()
  print("ALL TESTS PASSED")


if __name__ == "__main__":
  asyncio.run(main())
