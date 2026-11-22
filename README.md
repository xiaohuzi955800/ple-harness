# lin-ple-harness

An autonomous agent harness for Google's Kaggle competition
[Agent Harness on Physical-Engineering's Last Exam (PLE)](https://www.kaggle.com/competitions/agent-harness-on-physical-engineerings-last-exam):
it solves engineering tasks (structural, civil, electrical, mechanical) by
planning, executing with tools, and independently verifying its own
deliverables before the deadline.

Built on the official `custom-openai` sample, plus a three-role structure
(planner / worker / verifier) and an engineering layer: an evidence ledger,
hard-constraint-first prompting, deliverable verification, and fix rounds.

## How it works

- **Plan** (read-only, first 10% of the task time): reads the task, checks
  the hard constraints, and writes a short plan.
- **Work**: executes the plan with the full tool set and MCP servers,
  keeping an evidence ledger (`/tmp/evidence.md`) of every claim and its
  source. Hard constraints come first — a compliant deliverable is produced
  early, then optimized.
- **Verify** (fresh read-only agent): checks the deliverables against the
  task and the evidence ledger, and reports PASS or numbered gaps.
- **Fix**: up to 2 rounds of fixes for gaps found by the verifier.

The target model is `google-cloud/gemini-4-argon` (see `submission.yaml`).
Local runs use a Google AI Studio key via the kit's `gemini/<model>` path
(see `.github/workflows/smoke.yml`).

## Layout

- `serve.py` — the harness server (plan / work / verify / fix).
- `taskbridge.py` — serve protocol implementation (from the official sample).
- `submission.yaml` — entry declaration (model, install, serve, env).
- `requirements.txt` — Python dependencies.
- `tests/test_harness.py` — offline checks (39 checks, no model calls).
- `.github/workflows/smoke.yml` — GitHub Actions smoke test: `kit check`
  plus one dev task.

## License

Apache-2.0 (see `LICENSE`).
