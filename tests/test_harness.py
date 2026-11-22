"""Tests for the plan-work-verify harness (no model calls, no Docker).

Run from the harness directory:

  uv run --with openai==3.26.1 --with websockets==15.0.1 \\
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
        FakeAgent.first("verifier").context == "worker summary")
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
  check("skip, no files", await serve._ifc_gate(worker) == "")
  check("pass", await serve._ifc_gate(worker) == "")
  gaps = await serve._ifc_gate(worker)
  check("violations returned", "3 EXPRESS violations" in gaps, gaps)
  check("tool error is inconclusive",
        await serve._ifc_gate(worker) == "")

  class RaisingWorker:
    def __init__(self, session):
      self.session = session

    async def bash(self, command, timeout_sec=120):
      raise RuntimeError("boom")

  check("bash raising is inconclusive",
        await serve._ifc_gate(RaisingWorker(session)) == "")
  check("gate script compiles",
        compile(serve.IFC_GATE_SCRIPT, "<ifc-gate>", "exec") is not None)
  check("gate command is a heredoc",
        serve.IFC_GATE_COMMAND.startswith("python3 - <<'PYEOF'\n"))


async def test_gate_blocks_pass_then_fix():
  print("IFC gate turns a verifier PASS into a fix round")
  FakeAgent.reset(
      scripts={
          "planner": ["p"],
          "worker": ["worker summary"],
          "verifier": ["PASS", "PASS"],
      },
      bash_outputs=[
          "exit code 0\nIFC output/model.ifc: 1 EXPRESS violations\n"
          "  bad extrusion\nIFC_GATE: FAIL",
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
  await test_gate_blocks_pass_then_fix()
  test_prompts()
  print("ALL TESTS PASSED")


if __name__ == "__main__":
  asyncio.run(main())
