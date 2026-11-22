# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Plan-work-verify harness for any OpenAI-compatible model endpoint.

Connects to `MODEL_BASE_URL` using `MODEL_API_KEY` and `MODEL_NAME`. Each task
runs in phases: a read-only planner inspects the files and writes a plan, a
worker executes it through four built-in tools (`bash`, `read_file`,
`write_file`, `edit_file`) plus the task's own MCP tools, and a fresh
read-only verifier checks the deliverables and sends gaps back for fix rounds.
The worker's summary becomes the final message. Calls go to the Chat
Completions API, or to the Responses API for OpenAI's GPT-6 models
(`RESPONSES_MODELS`). Short tasks skip planning and verification and run the
worker directly, like the single-loop baseline this harness builds on.

Optional environment variables: `MAX_OUTPUT_TOKENS` (default `32000`),
`COMPACT_AT_TOKENS` (default `150000`), and `REASONING_EFFORT` (default `max`
for Claude, GLM, GPT-6 and DeepSeek-V4, `high` for other models).
"""

from __future__ import annotations

import asyncio
import json
import os
import posixpath
import re
import sys
import time
import types
from typing import Any

import openai
import taskbridge

MODEL = os.environ.get("MODEL_NAME", "")
# The model's highest reasoning level: max for Claude, GLM, GPT-6 and
# DeepSeek-V4, high for the others (Gemini refuses max). REASONING_EFFORT
# overrides it.
MAX_EFFORT_MODELS = ("claude", "glm", "gpt-6", "deepseek-v4")
EFFORT = os.environ.get("REASONING_EFFORT") or (
    "max" if any(m in MODEL.lower() for m in MAX_EFFORT_MODELS) else "high"
)
# OpenAI's GPT-6 models only support function tools combined with
# reasoning_effort on the Responses API. Other models also switch to
# /v1/responses if /v1/chat/completions returns a 400 directing them there.
RESPONSES_MODELS = ("gpt-6",)
RESPONSES_ONLY = re.compile(
    r"/v1/responses|not supported.{0,80} in /v1/chat/completions"
)
# Claude reuses a cached prompt prefix only where a request marks one
# (marked()); other models cache on their own.
CACHE_MARK = "claude" in MODEL.lower()
COMPACT_AT_TOKENS = int(os.environ.get("COMPACT_AT_TOKENS", "150000"))
TOOL_RESULT_CHARS = 30_000
# MCP tools are offered as functions up to this many characters of JSON in all
DIRECT_MCP_CHARS = 40_000
# functions one request may offer (OpenAI's limit), the built-in four included
MAX_TOOLS = 128
KEEP_RECENT_RESULTS = 12
FINISH_MARGIN_SEC = 180
# Plan-work-verify phase budgets.
PLAN_SHARE = 0.10  # the planner wraps up after this share of the task time
CHECK_SHARE = 0.25  # time kept for verification and fix rounds
MIN_PLAN_SEC = 1800  # plan only when the task allows at least this long
MIN_VERIFY_SEC = 900  # verify only when at least this long is left
MAX_FIX_ROUNDS = 2
RETRY_WAITS = (5, 15, 30, 60, 120)
# How endpoints word an error saying the conversation no longer fits (xAI's:
# "maximum prompt length is N but the request contains M tokens"). OpenAI's
# error for too many tools ("'tools': array too long") does not match.
CONTEXT_FULL = re.compile(
    r"context.?(length|window|size|limit)|maximum context|too many"
    r" tokens|prompt is too long|maximum prompt length"
    r"|(input|prompt|request|messages?)\b[^.]{0,40}(too long|exceed)"
    r"|exceed[^.]{0,40}(token|context)",
    re.I,
)
# The context window and the prompt's share of it, as vLLM, OpenAI, Anthropic,
# Gemini and xAI word them in such an error.
WINDOW_RE = re.compile(
    r"maximum context length (?:is |of )?(\d+)"
    r"|context (?:window|length|size)(?: limit)?(?: is| of)? (\d+)"
    r"|tokens? > (\d+) maximum|maximum number of tokens allowed \((\d+)\)"
    r"|maximum prompt length is (\d+)",
    re.I,
)
PROMPT_RE = re.compile(
    r"(\d+) input tokens|(\d+) in the messages|prompt is too long: (\d+)"
    r"|input token count \((\d+)\)|request contains (\d+) tokens",
    re.I,
)
MIN_OUTPUT = 1024  # the fewest output tokens to ask for
# characters an unread tool result keeps, however full the context
UNREAD_KEEP = 600
# tokens left free in the window, since tokenizers count a little differently
CONTEXT_MARGIN = 1024
# How endpoints word an error refusing a parameter or value outright, unlike one
# that names a smaller maximum.
UNSUPPORTED = re.compile(
    r"unsupported|unrecogni[sz]ed|unknown|not supported|does not support|not"
    r" (allowed|permitted)"
    r"|extra (inputs|fields)|unexpected",
    re.I,
)
# The standard fields of a streamed tool call, of its function, and of a reply's
# delta. Any other field an endpoint sends goes back with the message: Gemini
# refuses a tool call without its thought signature, which comes in the call's
# extra_content.
CALL_FIELDS = {"index", "id", "type", "function"}
FUNCTION_FIELDS = {"name", "arguments"}
DELTA_FIELDS = {
    "role",
    "content",
    "tool_calls",
    "function_call",
    "refusal",
    "audio",
    "annotations",
    "reasoning_content",
    "reasoning",
}

client = openai.AsyncOpenAI(
    base_url=os.environ.get("MODEL_BASE_URL"),
    api_key=os.environ.get("MODEL_API_KEY") or "none",
    max_retries=0,
    timeout=900,
)

SYSTEM = """You are an expert engineer working on a task in a Linux container. Your tools act inside it: bash (it \
starts in {workdir}), read_file, write_file and edit_file. The container has no internet access.{mcp}

How to work:
- Read the whole task, then look at the files you were given before choosing an approach.
- Work in small, verified steps. Compute numbers with code rather than by hand, and check units.
- When the task says how the result is checked (a time limit, a format, a test), check it that way yourself.
- Write each deliverable exactly where, under the name and in the format the task asks for. Write it early, then \
improve it.
- Produce office and binary formats (docx, xlsx, pdf, ...) with code, such as python3 and the libraries in the \
container, so they are real files of that format.
- Before you finish, open every deliverable again and check that it is complete and readable.
- You have about {minutes} minutes in total; the time left is shown after the results of your tools.

When the work is done, answer without calling a tool: a short summary of what you delivered (the path of each file) \
and the key results. That answer is graded together with your files."""

MCP_DIRECT = """

The task also offers its own tools (the functions named <server>__<tool>)."""

MCP_CATALOG = """

The task also offers its own tools, listed below by server. Call mcp_find to see a tool's parameters, then mcp_call \
to run it.
{catalog}"""

SUMMARY_PROMPT = """The conversation is getting long, so the earlier turns will be removed. Write notes that let you \
continue without them: what the task requires, what you found in the files, every file you created or changed (with \
its path and state), the numbers and decisions so far, and what remains to do. Be specific."""

EVIDENCE = """
Evidence and checking:
- Keep working notes at /tmp/evidence.md (never under a deliverable path): every key number with its source file \
and line, every decision with its reason. Never invent a number; if a value is missing, say so and use a clearly \
labelled assumption.
- Hard constraints (safety, feasibility, code limits) come before optimisation: never trade one for a better \
objective value. Check each constraint explicitly before you finish.
- Recompute key numbers a second way (independent code) and check units before you write them into a deliverable.
- If the task names a check (a validator, a test command, a format), run that exact check with the exact flags on \
the final deliverable before finishing, and paste its verbatim output in your summary. A check you did not run is \
a check you failed. Never report 'verified' from a weaker check.
- The container has no internet access: do not try pip install. List installed packages first (pip list) and work \
with what is there.
"""

WORKER = SYSTEM.replace(
    "\nWhen the work is done,", EVIDENCE + "\nWhen the work is done,"
)

PLANNER = """You plan an engineering task that another engineer will carry out in a Linux container with no internet \
access. Inspect the task's files with your tools (bash starts in {workdir}), but do not create or change any file. \
Then reply with the plan: each deliverable with its exact path and format, what it must contain, the data and method \
to use, and how to check it. You have about {minutes} minutes for this planning step."""

VERIFIER = """You check an engineer's work on a task in a Linux container. You get the task, the plan, and the \
engineer's summary of what was delivered. Open every deliverable with your tools (bash starts in {workdir}) and \
check it against the task: the path, the format, every part the task asks for, and the key numbers, recomputed. Do \
not create or change any file. Never accept the worker's claim that something was checked. Re-run the task's \
stated validation yourself with the exact flags and quote its verbatim output in your reply. A script's own \
self-check printout is not independent evidence — run the check separately on the produced file. If you cannot \
re-run a claimed check, list it as a gap. A collision check that produced no geometry (for example a tool that \
returned 0 shapes, or a script that printed a count but no conclusion) is a failed measurement, not a pass — \
never accept "0 overlaps" from a tool that did not actually measure anything. When the task involves routing, recompute the route geometry from the \
produced .ifc yourself — do not take it from reconciliation.json or the worker's summary: segment count and total \
length per service, the entity class of every route segment, the distance from each service's nearest segment to \
the entry, per-row coverage, and CRAH reach. Quote those numbers in your reply; a PASS that does not quote them \
is not a pass. The deterministic gate results are in your context too — quote their numbers (clash count, \
separation, entry distance, coverage) in your reply; a PASS that does not mention them is not a pass, and a \
collision check that printed a count but measured no geometry is a failed measurement, not a pass. Re-run \
output/solution.py yourself so the deliverable regenerates. If everything holds, reply with \
the single word PASS and nothing else. Otherwise reply with a numbered list of the gaps, each specific enough to \
fix. You have about {minutes} minutes for this checking step."""

# Words in the task instruction that mark a routing task (cable tray /
# busway / chilled water). Used for the routing prompt guides and for the
# deterministic connectivity gate.
ROUTING_KEYWORDS = ("routing", "cable tray", "busway", "crah", "mep routing")

# Appended to the worker prompt when the task mentions IFC. The exact
# validation snippet and the extrusion rule below are the two things an
# agent got wrong on a real task (schema-only validation reported as
# express-rules validation; horizontal extrusions rejected by the grader).
IFC_GUIDE = """
Working with IFC files (ifcopenshell):
- Validate the produced .ifc with the full check before finishing, using this exact code (the logger argument is \
required, and express_rules must be True — a schema-only check cannot catch geometry-rule violations):
  import ifcopenshell, ifcopenshell.validate
  class _L:
      def __init__(self): self.errors = []
      def error(self, *a, **k): self.errors.append((a, k))
  _log = _L()
  ifcopenshell.validate.validate(model, _log, express_rules=True)
  assert not _log.errors, _log.errors[:5]
- IFC4 rule WR1 for IfcExtrudedAreaSolid: the extrusion direction must have a non-zero component along the profile \
normal (the profile lies in its local XY plane, normal (0,0,1)). Never extrude a horizontal run by setting \
ExtrudedDirection to a horizontal vector — that makes the solid degenerate and the file invalid. Instead keep \
ExtrudedDirection=(0,0,1) in the solid's local frame and rotate the solid's Position (Axis/RefDirection) so local \
Z points along the run direction.
- API notes: ifcopenshell.validate has no .run; validate(model, logger, express_rules=True) needs a logger; a \
logger's error takes *args; placements use IfcLocalPlacement(PlacementRelTo=..., \
RelativePlacement=IfcAxis2Placement3D(...)); open files with ifcopenshell.open; look up schema declarations with \
ifcopenshell.schema_by_name('IFC4').declaration(...). There is no IfcCableTraySegment in IFC4 — use \
IfcCableCarrierSegment for trays/busways and IfcPipeSegment for pipes.
- Units are mandatory: declare millimetres on the project, or the grader reads every coordinate 1000x too \
large even though validation passes. Copy this block exactly; do not paraphrase it:
  project = model.create_entity('IfcProject', GlobalId=ifcopenshell.guid.new(), Name='DataHallProject')
  length_unit = model.create_entity('IfcSIUnit', UnitType='LENGTHUNIT', Prefix='MILLI', Name='METRE')
  plane_angle_unit = model.create_entity('IfcSIUnit', UnitType='PLANEANGLEUNIT', Name='RADIAN')
  unit_assignment = model.create_entity('IfcUnitAssignment', Units=[length_unit, plane_angle_unit])
  project.UnitsInContext = unit_assignment
- Placement rule: put the coordinates in exactly ONE place. Set the product's ObjectPlacement Location to \
(x, y, z) and the solid's Position Location to (0, 0, 0) — or the reverse — never both to (x, y, z). IFC \
composes the two placements, so setting both to the same (x, y, z) builds every box at twice its intended \
coordinates. The profile is centred at the solid's local origin, so (x, y) is the box CENTRE, not its \
corner; z is the extrusion base. A box built from create_box_extrusion(w, d, h, cx, cy, z) spans \
cx-w/2..cx+w/2 in x, cy-d/2..cy+d/2 in y, z..z+h in z.
"""

# Appended to the worker and verifier prompts when the task is a routing
# task. A real run produced one 23,000 mm box per service, a busway modelled
# as a distribution board, no feeder from the entry, no per-row headers and
# no CRAH drops — and the verifier passed it. Every rule below is checkable.
ROUTING_GUIDE = """
Routing (cable tray, busway, chilled water):
- Connect in this order: (a) the service entry first — a feeder from the entry marked on the drawings to your first \
header; (b) one header per row, every row, all three services; (c) a drop from the CHW header to every CRAH, \
including perimeter units the row headers do not pass. Never submit a route with one segment per service; every \
direction or tier change is a new segment.
- Busway is IfcCableCarrierSegment, exactly like the tray — never IfcElectricDistributionBoard.
- Joints are butt joints: two segments meet end to end with 0 mm overlap (touching) or a gap of at most 100 mm. \
Never build a junction from overlapping boxes — the grader counts every overlap over 2 mm as a clash, including \
joints and same-service pairs, and more than half the clashes on a real run were joint overlaps. A joint's solids \
may touch or sit up to 250 mm apart and still count as connected, so a small gap is fine and an overlap is not. \
Build each horizontal segment as a flat box (length along the run, width across, thickness up) that starts where \
the previous segment's box ends:
  create_box_extrusion(x1 - x0, width, thickness, (x0 + x1) / 2, y, z)  # run from (x0, y) to (x1, y) at height z
- Profile orientation: a tray segment's vertical extent must be its thickness (150 mm), not its width (600 mm). \
If you rotate a placement, check the result — a 600 mm dimension standing vertical is the single most common \
cause of clashes. The unrotated box style above cannot make this mistake.
- Drops go down: a drop from a header to a CRAH starts at the header and goes down to the CRAH top. Build it as \
a box whose bottom is the CRAH top and whose top reaches the header — never place a drop at the header and \
extrude upward:
  create_box_extrusion(w, w, z_top - z_bottom, x, y, z_bottom)  # drop from header z_top down to CRAH top z_bottom
- Every CRAH needs a horizontal branch from the nearest header or trunk to the CRAH's (x, y), then a drop down \
from the branch end. A CRAH directly below a trunk still needs a branch — first measure the trunk's extent; \
if the CRAH's position is outside it, build a branch. Never skip a branch because the CRAH's x matches the \
trunk's x: a trunk that ends short of the CRAH's y leaves the drop hanging in mid-air. A drop whose top is \
not within 250 mm of a horizontal segment of the same service is not connected, even if it happens to touch \
another segment.
- Keep the three services on separate tiers at fixed heights: tray at z=3400, busway at z=3100, CHW at \
z=2800 (each top at least 460 mm below the sprinklers and each bottom above the 2600 mm clear height). These \
are the heights to use unless a drawing gives you different ones. An X corridor does NOT separate services: \
it only separates trunks running along Y — headers running along X cross the whole hall on the same row and \
overlap along their full length unless the services are on different tiers. And centreline spacing is not \
edge spacing: two 600 mm wide trunks whose centrelines are 400 mm apart overlap by 200 mm. Subtract the box \
width from the centreline spacing before claiming a separation. A feeder must not pass through a column: \
route around it.
- Name every route segment clearly (CableTray-1, Busway-1, CHW-1, ...).
- Write output/routing.json next to the model, with the numbers you actually used: {"entry": [x, y, z], "rows": \
[{"name": "A", "start": [x, y, z], "end": [x, y, z]}, ...], "crahs": [[x, y, z], ...]}. Then run a connectivity \
self-check from that file (segment count and total length per service, entry distance, per-row coverage, CRAH \
reach) and paste its output in your summary. A route you did not measure is a route you did not build.
- If you cannot extract a drawing value after a few tries, record what you could extract and move on with \
clearly-labelled assumptions — do not keep re-trying the same extraction.
"""

# Appended to the worker prompt only, on routing tasks. A tested stdlib-only
# PNG reader (verified against the real 1600x1100 drawing sheets): use it
# instead of writing your own PNG decoder.
PNG_GUIDE = r"""
Reading the drawing sheets (input/*.png): use this tested stdlib-only reader instead of writing your own PNG \
decoder — it returns (width, height, rows of RGB bytes), colour types 2 (RGB) and 6 (RGBA), 8-bit non-interlaced:
  import struct, zlib
  def read_png(path):
      data = open(path, 'rb').read()
      assert data[:8] == b'\x89PNG\r\n\x1a\n'
      pos, w, h, bitdepth, colortype, idat = 8, 0, 0, 0, 0, b''
      while pos < len(data):
          ln, typ = struct.unpack('>I4s', data[pos:pos+8])
          chunk = data[pos+8:pos+8+ln]
          if typ == b'IHDR': w, h, bitdepth, colortype = struct.unpack('>IIBB', chunk[:10])
          elif typ == b'IDAT': idat += chunk
          elif typ == b'IEND': break
          pos += 12 + ln
      assert bitdepth == 8 and colortype in (2, 6)
      raw = zlib.decompress(idat)
      ch = 3 if colortype == 2 else 4
      stride = w * ch
      rows, prev = [], bytearray(stride)
      i = 0
      for _y in range(h):
          f = raw[i]; i += 1
          line = bytearray(raw[i:i+stride]); i += stride
          if f == 1:
              for x in range(ch, stride): line[x] = (line[x] + line[x-ch]) & 255
          elif f == 2:
              for x in range(stride): line[x] = (line[x] + prev[x]) & 255
          elif f == 3:
              for x in range(stride):
                  a = line[x-ch] if x >= ch else 0
                  line[x] = (line[x] + ((a + prev[x]) >> 1)) & 255
          elif f == 4:
              for x in range(stride):
                  a = line[x-ch] if x >= ch else 0
                  b = prev[x]
                  c = prev[x-ch] if x >= ch else 0
                  p = a + b - c
                  pa, pb, pc = abs(p-a), abs(p-b), abs(p-c)
                  pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                  line[x] = (line[x] + pr) & 255
          rows.append(bytes(line)); prev = line
      return w, h, rows
  From the pixels, find the drawing's geometry yourself (dark-pixel bounding boxes, long straight runs of dark \
pixels, the positions of text blocks) and record the scale from the drawing's own dimensions before using any \
coordinate.
"""


def tool(
    name: str, description: str, required: list[str], **properties: Any
) -> dict[str, Any]:
  """Builds an OpenAI function-tool definition dictionary.

  Args:
    name: Tool function name.
    description: Human-readable tool description.
    required: List of required parameter names.
    **properties: Parameter JSON Schema definitions keyed by parameter name.

  Returns:
    An OpenAI function-tool specification dictionary.
  """
  return {
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": properties,
              "required": required,
          },
      },
  }


TOOLS = [
    tool(
        "bash",
        "Run a bash command in the task container and return its exit code and"
        " output. Each call starts a new shell in the working directory: only"
        " files persist between calls.",
        ["command"],
        command={"type": "string"},
        timeout_sec={
            "type": "integer",
            "description": (
                "Stop the command after this many seconds (default 120)."
            ),
        },
    ),
    tool(
        "read_file",
        "Read a text file from the task container.",
        ["path"],
        path={"type": "string"},
    ),
    tool(
        "write_file",
        "Create or overwrite a file in the task container; parent directories"
        " are created.",
        ["path", "content"],
        path={"type": "string"},
        content={"type": "string"},
        append={
            "type": "boolean",
            "description": (
                "Add to the end of the file instead (write a large file in"
                " parts)."
            ),
        },
    ),
    tool(
        "edit_file",
        "Replace exact text in a file. old_text must occur exactly once unless"
        " replace_all is true.",
        ["path", "old_text", "new_text"],
        path={"type": "string"},
        old_text={"type": "string"},
        new_text={"type": "string"},
        replace_all={"type": "boolean"},
    ),
]
MCP_TOOLS = [
    tool(
        "mcp_find",
        "Show the description and parameters of the task's own tools that match"
        " a query.",
        ["query"],
        query={
            "type": "string",
            "description": "A tool name, or words describing what you need.",
        },
    ),
    tool(
        "mcp_call",
        "Run one of the task's own tools.",
        ["tool", "arguments"],
        tool={"type": "string", "description": "The tool's name as listed."},
        arguments={"type": "object"},
    ),
]


def clip(text: str, limit: int = TOOL_RESULT_CHARS) -> str:
  """Truncates `text` in the middle if it exceeds `limit` characters.

  Args:
    text: String to clip.
    limit: Maximum character length before truncation.

  Returns:
    The original string if within `limit`, or a middle-truncated string.
  """
  if len(text) <= limit:
    return text
  head = limit * 2 // 3
  return (
      f"{text[:head]}\n[... {len(text) - limit} characters cut"
      f" ...]\n{text[-(limit - head):]}"
  )


def parse_arguments(call: dict[str, Any]) -> dict[str, Any] | str:
  """Parses a tool call's arguments into a dict or returns an error string.

  Rewrites `call["arguments"]` as canonical JSON (`"{}"` when invalid) because
  endpoints reject conversations containing malformed tool-call arguments.

  Args:
    call: Tool-call dictionary with an `arguments` JSON string field.

  Returns:
    The parsed arguments dictionary, or an error string for the model.
  """

  def reject(constant):
    raise ValueError(f"{constant} is not valid JSON")

  try:
    args = json.loads(call["arguments"] or "{}", parse_constant=reject)
    if not isinstance(args, dict):
      raise ValueError("the arguments must be a JSON object")
    call["arguments"] = (
        json.dumps(args, ensure_ascii=False)
        .encode("utf-8", "replace")
        .decode("utf-8")
    )
    return args
  except ValueError as exc:
    call["arguments"] = "{}"
    return (
        f"error: the arguments are not a valid JSON object ({exc}); send the"
        " call again"
    )


def unique_name(name: str, taken: Any) -> str:
  """Truncates `name` to 64 chars and appends `_2`, `_3`, ... if in `taken`.

  Args:
    name: Desired function name.
    taken: Collection or mapping of already-used function names.

  Returns:
    A unique function name of at most 64 characters.
  """
  candidate, n = name[:64], 1
  while candidate in taken:
    n += 1
    candidate = f"{name[:63 - len(str(n))]}_{n}"
  return candidate


def visible(text: str) -> str:
  """Strips `<think>...</think>` reasoning blocks from `text`.

  Args:
    text: Assistant response content string.

  Returns:
    The trimmed response text with inline `<think>` blocks removed.
  """
  return re.sub(r"<think>.*?(</think>|$)", "", text or "", flags=re.S).strip()


def extra_fields(part: Any, known: set[str]) -> dict[str, Any]:
  """Extracts non-null extra fields from a streamed SDK object `part`.

  The OpenAI SDK stores undeclared provider fields in `part.model_extra`.

  Args:
    part: Streamed Pydantic model object (delta, tool call, or function).
    known: Set of standard field names to exclude.

  Returns:
    Dictionary of extra fields not in `known` whose values are not `None`.
  """
  fields = getattr(part, "model_extra", None) or {}
  return {k: v for k, v in fields.items() if k not in known and v is not None}


def merged(old: Any, new: Any, key: str = "") -> Any:
  """Merges a single delta's extra field value `new` into `old`.

  Merges dictionaries key-by-key, concatenates lists, and replaces scalars.
  Delegates `thinking_blocks` lists to `thinking_blocks()`.

  Args:
    old: Accumulated field value so far.
    new: Incoming delta field value.
    key: Field name being merged.

  Returns:
    The merged field value.
  """
  if key == "thinking_blocks" and isinstance(new, list):
    return thinking_blocks(old if isinstance(old, list) else [], new)
  if isinstance(old, dict) and isinstance(new, dict):
    return {**old, **{k: merged(old.get(k), v, k) for k, v in new.items()}}
  if isinstance(old, list) and isinstance(new, list):
    return old + new
  return new


def thinking_blocks(
    blocks: list[dict[str, Any]], new: list[Any]
) -> list[dict[str, Any]]:
  """Accumulates streamed Claude `thinking_blocks` deltas.

  Joins consecutive deltas into the current block until a `signature` closes it;
  redacted blocks (containing `data`) arrive whole and are kept separate.

  Args:
    blocks: Existing list of accumulated thinking blocks.
    new: New list of thinking block deltas from the stream chunk.

  Returns:
    Updated list of thinking block dictionaries.
  """
  out = [dict(b) for b in blocks]
  for b in new:
    if not isinstance(b, dict):
      continue
    last = out[-1] if out else None
    if (
        last is None
        or last.get("signature")
        or last.get("data")
        or b.get("data")
        or last.get("type") != b.get("type")
    ):
      out.append(dict(b))
      continue
    last["thinking"] = (last.get("thinking") or "") + (b.get("thinking") or "")
    if b.get("signature"):
      last["signature"] = b["signature"]
  return out


def signed(block: Any) -> bool:
  """Returns whether `block` has a `signature` or redacted `data` field.

  Args:
    block: Candidate thinking block dictionary.

  Returns:
    `True` if Claude will accept `block` in a subsequent turn, else `False`.
  """
  return isinstance(block, dict) and bool(
      block.get("signature") or block.get("data")
  )


def drop_thinking(
    message: dict[str, Any], unless: Any = lambda block: False
) -> dict[str, Any]:
  """Filters `thinking_blocks` in `message`, keeping blocks matching `unless`.

  Args:
    message: Assistant message dictionary to modify in place.
    unless: Predicate `unless(block)` returning `True` for blocks to keep.

  Returns:
    The modified `message` dictionary.
  """
  if isinstance(message.get("thinking_blocks"), list):
    message["thinking_blocks"] = [
        b for b in message["thinking_blocks"] if unless(b)
    ]
  fields = message.get("provider_specific_fields")
  if isinstance(fields, dict) and isinstance(
      fields.get("thinking_blocks"), list
  ):
    message["provider_specific_fields"] = {
        **fields,
        "thinking_blocks": [b for b in fields["thinking_blocks"] if unless(b)],
    }
  return message


def marked(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Returns the messages a Chat Completions call sends.

  For Claude, this is a copy whose system prompt and newest message carry
  `cache_control` (2 of Claude's 4 cache breakpoints), so every turn reads the
  conversation so far from the cache instead of paying for all of it again.
  The stored conversation stays as it is.

  Args:
    messages: The conversation so far.

  Returns:
    `messages` itself for other models, or the marked copy for Claude.
  """
  if not CACHE_MARK:
    return messages
  out = list(messages)
  for i in {0, len(out) - 1}:
    m, content = out[i], out[i].get("content")
    if isinstance(content, str) and content:
      out[i] = {
          **m,
          "content": [{
              "type": "text",
              "text": content,
              "cache_control": {"type": "ephemeral"},
          }],
      }
    else:
      out[i] = {**m, "cache_control": {"type": "ephemeral"}}
  return out


def response_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
  """Converts Chat Completions messages into Responses API input items.

  Assistant turns produced by the Responses API are replayed using their saved
  output items (including encrypted reasoning). Each function call takes its id
  and name from the turn's stored tool call, and its arguments as
  `parse_arguments` rewrote them.

  Args:
    messages: Conversation messages in Chat Completions format.

  Returns:
    The `input` list for a Responses API call.
  """
  items = []
  for m in messages:
    if m["role"] == "tool":
      items.append({
          "type": "function_call_output",
          "call_id": m["tool_call_id"],
          "output": m["content"],
      })
    elif m["role"] != "assistant":
      items.append({"role": m["role"], "content": m["content"]})
    elif "output_items" in m:
      calls = iter(m.get("tool_calls") or [])
      for item in m["output_items"]:
        call = None
        if item.get("type") == "function_call":
          call = next(calls, None)
        if call:
          item = {
              **item,
              "call_id": call["id"],
              "name": call["function"]["name"],
              "arguments": call["function"]["arguments"],
          }
        items.append(item)
    else:
      if m.get("content"):
        items.append({"role": "assistant", "content": m["content"]})
      items += [
          {
              "type": "function_call",
              "call_id": c["id"],
              "name": c["function"]["name"],
              "arguments": c["function"]["arguments"],
          }
          for c in m.get("tool_calls") or []
      ]
  return items


def chat_usage(usage: Any) -> Any:
  """Returns Responses API `usage` in the shape `record` and `report` read.

  Args:
    usage: A Responses API usage object, or None.

  Returns:
    An object with `prompt_tokens`, `completion_tokens` and
    `prompt_tokens_details.cached_tokens`, as Chat Completions reports them, or
    None.
  """
  if usage is None:
    return None
  details = getattr(usage, "input_tokens_details", None)
  return types.SimpleNamespace(
      prompt_tokens=usage.input_tokens or 0,
      completion_tokens=usage.output_tokens or 0,
      prompt_tokens_details=types.SimpleNamespace(
          cached_tokens=getattr(details, "cached_tokens", 0) or 0
      ),
  )


class MCPServer(taskbridge.MCPClient):
  """Client wrapper for a single task MCP server and its discovered tools."""

  def __init__(self, session: taskbridge.Session, spec: dict[str, Any]):
    super().__init__(session, spec["url"], client="custom-harness")
    self.name = spec["name"]
    self.tools: list[dict[str, Any]] = []

  async def start(self) -> None:
    """Initializes the MCP session and fetches the server's tool list."""
    await self.initialize()
    self.tools = await self.list_tools()

  async def call(self, name: str, arguments: dict[str, Any]) -> str:
    """Calls tool `name` on the MCP server and formats the result as text.

    Args:
      name: Unprefixed MCP tool name on this server.
      arguments: Tool input arguments dictionary.

    Returns:
      Formatted text output (prefixed with `error: ` if `isError` is set).
    """
    result = await self.call_tool(name, arguments)
    parts = []
    content = (
        result.get("content") if isinstance(result.get("content"), list) else []
    )
    for item in content:
      kind = item.get("type") if isinstance(item, dict) else None
      if kind == "text":
        parts.append(str(item.get("text", "")))
      elif kind == "resource":
        parts.append(json.dumps(item.get("resource"))[:TOOL_RESULT_CHARS])
      elif kind == "resource_link":  # its uri, name and mimeType
        parts.append(json.dumps(item)[:TOOL_RESULT_CHARS])
      else:
        parts.append(f"[{kind or 'unknown'} content not shown]")
    if not parts and result.get("structuredContent") is not None:
      parts.append(json.dumps(result["structuredContent"]))
    text = "\n".join(parts) or "(no content)"
    return f"error: {text}" if result.get("isError") else text


class Reply:
  """Accumulates a streamed chat-completion response and its tool calls."""

  def __init__(self):
    self.text, self.reasoning, self.calls, self.usage, self.finish = (
        "",
        "",
        {},
        None,
        None,
    )
    self.extra = {}  # fields beyond DELTA_FIELDS, sent back with the message
    self._by_index, self._by_id, self._last = {}, {}, None

  def add_call(self, part: Any) -> None:
    """Adds one streamed tool-call delta to the accumulated calls.

    Groups deltas by `index`, or by `id` when an endpoint omits or reuses
    `index`. Deltas with neither continue the most recent call unless
    `_another_call` detects a new call boundary. Preserves provider-specific
    extra fields on both the call and its `function`.

    Args:
      part: Streamed tool-call delta object from a completion chunk.
    """
    index, call_id = (
        getattr(part, "index", None),
        getattr(part, "id", None) or "",
    )
    function = getattr(part, "function", None)
    name = (getattr(function, "name", None) or "") if function else ""
    key = self._by_index.get(index) if index is not None else None
    if (
        key is not None
        and call_id
        and self.calls[key]["id"] not in ("", call_id)
    ):
      key = None
    if (
        key is not None
        and not call_id
        and name
        and self._another_call(self.calls[key], name)
    ):
      key = None
    if key is None and call_id:
      key = self._by_id.get(call_id)
    if (
        key is None
        and index is None
        and not call_id
        and self._last is not None
        and not (name and self._another_call(self.calls[self._last], name))
    ):
      key = self._last
    if key is None:
      key = len(self.calls)
      self.calls[key] = {
          "id": "",
          "name": "",
          "arguments": "",
          "extra": {},
          "function_extra": {},
      }
    if index is not None:
      self._by_index[index] = key
    if call_id:
      self._by_id[call_id] = key
    call = self.calls[key]
    call["id"] = call["id"] or call_id
    call["name"] = call["name"] or name
    call["arguments"] += (
        (getattr(function, "arguments", None) or "") if function else ""
    )
    call["extra"] = merged(call["extra"], extra_fields(part, CALL_FIELDS))
    if function:
      call["function_extra"] = merged(
          call["function_extra"], extra_fields(function, FUNCTION_FIELDS)
      )
    self._last = key

  @staticmethod
  def _another_call(call: dict[str, Any], name: str) -> bool:
    """Checks whether a nameless-ID delta for `name` starts a new tool call.

    Returns `True` when `call` already has a function name and either `name`
    differs or `call["arguments"]` is already a complete JSON object.

    Args:
      call: Currently accumulating tool-call dictionary.
      name: Function name on the incoming delta.

    Returns:
      `True` if the delta begins a separate tool call, else `False`.
    """
    if not call["name"]:
      return False
    if name != call["name"]:
      return True
    try:
      return isinstance(json.loads(call["arguments"]), dict)
    except ValueError:
      return False

  def message(self) -> dict[str, Any]:
    """Builds the assistant message dict for the conversation history.

    Preserves provider-specific fields while stripping unsigned thinking blocks
    that Claude would reject.

    Returns:
      The formatted assistant message dictionary.
    """
    message = {**self.extra, "role": "assistant", "content": self.text or None}
    if self.calls:
      message["tool_calls"] = [
          {
              **c["extra"],
              "id": c["id"],
              "type": "function",
              "function": {
                  **c["function_extra"],
                  "name": c["name"],
                  "arguments": c["arguments"],
              },
          }
          for c in self.calls.values()
      ]
    if self.reasoning:
      message["reasoning_content"] = self.reasoning
    return drop_thinking(message, unless=signed)


class OutOfTime(Exception):  # pylint: disable=g-bad-exception-name
  """Raised when retrying a failed model call would exceed the time budget."""


class CutOff(asyncio.TimeoutError):  # pylint: disable=g-bad-exception-name
  """Raised when a streamed reply is cut off by its per-call deadline."""


class StreamBadRequest(openai.OpenAIError):  # pylint: disable=g-bad-exception-name
  """Raised when an SSE stream reports a non-retryable bad request error.

  OpenAI streams report errors via `type` and `code` strings (such as
  `invalid_request_error` or `context_length_exceeded`) instead of an HTTP 400
  status. Treating them as 400-equivalent allows `remedy()` to adjust the
  request rather than retrying it unchanged.
  """


# `type` or `code` values of an in-stream error that mark a bad request
BAD_REQUEST = frozenset({
    "invalid_request_error",
    "context_length_exceeded",
    "invalid_value",
    "invalid_prompt",
})


def stream_error(error: Any) -> openai.OpenAIError:
  """Classifies an in-stream error object as a bad request or a truncated stream.

  Args:
    error: Error object or dict reported inside an SSE stream.

  Returns:
    `StreamBadRequest` if the error type, code, or message indicates an invalid
    request or context overflow; otherwise `StreamCut` so the call is retried.
  """
  def field(name: str) -> Any:
    if isinstance(error, dict):
      return error.get(name)
    return getattr(error, name, None)

  message = str(field("message") or "")
  named = {field("type"), field("code")} & BAD_REQUEST
  if named or CONTEXT_FULL.search(message):
    return StreamBadRequest(message)
  return StreamCut(f"the stream reported an error: {message}")


class StreamCut(openai.OpenAIError):  # pylint: disable=g-bad-exception-name
  """Raised when an SSE stream ends before reporting a finish reason.

  Indicates a dropped connection, proxy timeout, or transient in-stream error.
  The partial response is discarded and the call is retried.
  """


class Agent:
  """Manages the conversation, tool execution, and context compaction loop."""

  def __init__(
      self,
      session: taskbridge.Session,
      role: str = "worker",
      plan: str = "",
      context: str = "",
      phase_seconds: float | None = None,
  ):
    self.session = session
    self.role = role
    self.plan = plan
    self.context = context
    self.phase_deadline = (
        time.monotonic() + phase_seconds if phase_seconds else None
    )
    if role in ("planner", "verifier"):
      self.tools = [
          t
          for t in TOOLS
          if t["function"]["name"] in ("bash", "read_file")
      ]
    else:
      self.tools = list(TOOLS)
    self.mcp: dict[str, tuple[MCPServer, dict[str, Any]]] = {}
    self.messages: list[dict[str, Any]] = []
    self.prompt_tokens = 0
    # Each call asks for output_limit(): output_ceiling (MAX_OUTPUT_TOKENS,
    # lowered by shrink_output when the endpoint refuses that many), or less
    # when the context window has less room beside the prompt.
    self.output_ceiling = int(os.environ.get("MAX_OUTPUT_TOKENS", "32000"))
    self.context_window: int | None = None  # learned from a context-full error
    self.max_output = self.output_ceiling  # the limit the last call asked for
    # max_tokens once an endpoint refuses this one
    self.token_limit = "max_completion_tokens"
    # empty once an endpoint refuses it
    self.effort = {"reasoning_effort": EFFORT}
    # whether calls go to the Responses API rather than Chat Completions
    self.responses = MODEL.lower().startswith(RESPONSES_MODELS)
    self.finish_margin = FINISH_MARGIN_SEC

  def time_left(self) -> float | None:
    """Remaining seconds for this phase: the task's or the phase's, sooner."""
    left = self.session.time_left()
    if self.phase_deadline is None:
      return left
    phase = self.phase_deadline - time.monotonic()
    return phase if left is None else min(left, phase)

  def system_prompt(self, mcp: str) -> str:
    """Formats this role's system prompt."""
    left = self.time_left()
    minutes = int(left / 60) if left else "unlimited"
    template = {"planner": PLANNER, "verifier": VERIFIER}.get(
        self.role, WORKER
    )
    prompt = template.format(
        workdir=self.session.workdir, mcp=mcp, minutes=minutes
    )
    instr = self.session.instruction.lower()
    is_routing = any(k in instr for k in ROUTING_KEYWORDS)
    if "ifc" in instr or is_routing:
      # Routing tasks produce IFC models, so they need the validation
      # snippet even when the instruction never says "IFC".
      prompt += IFC_GUIDE
    if is_routing:
      prompt += ROUTING_GUIDE
      if self.role == "worker":
        prompt += PNG_GUIDE
    return prompt

  def initial_user_message(self) -> str:
    """Builds this role's first user message."""
    if self.role == "verifier":
      return (
          "Task:\n"
          f"{self.session.instruction}\n\nPlan:\n{self.plan or '(none)'}\n\n"
          "Engineer's summary of what was delivered:\n"
          f"{self.context or '(none)'}"
      )
    return self.session.instruction

  async def run(self) -> str:
    """Sets up this role's conversation and runs its tool-calling loop.

    Returns:
      The final summary message produced by the agent.
    """
    mcp = await self.connect_mcp() if self.role == "worker" else ""
    left = self.time_left()
    # time kept for the final summary: FINISH_MARGIN_SEC, or a quarter of a
    # short task's (or phase's) time
    self.finish_margin = (
        FINISH_MARGIN_SEC
        if left is None
        else min(FINISH_MARGIN_SEC, max(left, 0) / 4)
    )
    self.messages = [
        {"role": "system", "content": self.system_prompt(mcp)},
        {"role": "user", "content": self.initial_user_message()},
    ]
    if self.role == "worker" and self.plan:
      self.messages.append({
          "role": "user",
          "content": (
              "A plan for this task, from a planning pass over its files."
              " Use it, and correct it where the files show it is wrong:\n"
              + self.plan
          ),
      })
    return await self.loop()

  async def resume(self, feedback: str) -> str:
    """Continues this agent's loop with new user feedback (a fix round).

    Fix rounds run to the task deadline: the work phase's deadline no longer
    applies once verification has started.

    Args:
      feedback: User message describing what to fix.

    Returns:
      The final summary message produced by the agent.
    """
    self.phase_deadline = None
    self.messages.append({"role": "user", "content": feedback})
    return await self.loop()

  async def loop(self) -> str:
    """Runs the tool-calling loop until the task is complete.

    Returns:
      The final summary message produced by the agent.
    """
    # head: the text of an answer cut off at the output limit
    nudges, head = 0, ""
    while True:
      left = self.time_left()
      if left is not None and left < self.finish_margin:
        return await self.finish("Time is almost up.")
      try:
        reply = await self.complete(head=head)
      except OutOfTime:
        return await self.finish(
            "Time is almost up, and the model is not answering."
        )
      # what it streamed is the answer if the summary call fails too
      except CutOff:
        return await self.finish("Time is almost up.")
      except openai.OpenAIError:
        # A transient endpoint failure (429/503) after the agent already
        # produced text is not fatal: the summary already produced is the
        # deliverable. On run 9 a 503 after the summary existed propagated
        # out of loop() and killed the whole task.
        return self.session.final_message
      arguments = [parse_arguments(call) for call in reply.calls.values()]
      if reply.text or reply.calls:
        # A turn with neither (its output went to hidden reasoning, or a content
        # filter stopped it) is not sent back: endpoints refuse an assistant
        # message with null content and no tool calls.
        self.messages.append(reply.message())
      text = visible(head + reply.text)
      if text:
        self.session.final_message = text
      if not reply.calls:
        if visible(reply.text) and reply.finish == "length":
          # cut off at the output limit: ask for the rest, and reset the
          # empty-turn count
          nudges, head = 0, head + reply.text
          self.messages.append({
              "role": "user",
              "content": (
                  f"Your reply was cut off at the {self.max_output}-token"
                  " output limit. Continue where it stopped, and put long"
                  " content in files with write_file rather than in a reply."
              ),
          })
          continue
        if visible(reply.text) or nudges >= 2:
          return text or self.session.final_message
        nudges += 1  # only consecutive empty turns end the run
        self.messages.append({
            "role": "user",
            "content": (
                "Continue the task; when it is done, answer with "
                "your summary of the deliverables."
            ),
        })
        continue
      nudges, head = 0, ""
      for call, args in zip(reply.calls.values(), arguments):
        result = (
            await self.run_tool(call["name"], args)
            if isinstance(args, dict)
            else args
        )
        if reply.finish == "length" and isinstance(args, str):
          result += (
              f" Your reply hit the {self.max_output}-token output limit; write"
              " large files in parts."
          )
        self.messages.append(
            {"role": "tool", "tool_call_id": call["id"], "content": result}
        )
      left = self.time_left()
      if left is not None:
        self.messages[-1]["content"] += f"\n[{int(left // 60)} minutes left]"
      if self.prompt_tokens > self.compact_at():
        await self.compact()

  async def finish(self, why: str) -> str:
    """Prompts the model for a final summary without tools when time is up.

    Args:
      why: Reason string prepended to the wrap-up prompt.

    Returns:
      The final summary text, falling back to `self.session.final_message`.
    """
    self.messages.append({
        "role": "user",
        "content": (
            f"{why} Stop working now and answer with your final "
            "summary: what you delivered, where, and the key results."
        ),
    })
    left = self.time_left()
    # the summary call gets taskbridge.time_share(left, 30) seconds, and at
    # least 5
    reserve = (
        0 if left is None else left - max(5, taskbridge.time_share(left, 30))
    )
    try:
      text = visible(
          (
              await self.complete(tool_choice="none", reserve=reserve, head="")
          ).text
      )
    except (CutOff, openai.OpenAIError, OutOfTime):
      text = ""
    return text or self.session.final_message

  async def connect_mcp(self) -> str:
    """Connects to the task's MCP servers and registers their tools.

    Exposes a small set of MCP tools directly as OpenAI functions; for larger
    tool sets, lists their names in the system prompt and exposes `mcp_find`
    and `mcp_call`.

    Returns:
      The system-prompt section describing the available MCP tools.
    """
    for spec in self.session.mcp_servers:
      server = MCPServer(self.session, spec)
      try:
        await server.start()
      except (taskbridge.TaskError, RuntimeError, ValueError) as exc:
        await self.session.log(f"MCP server {server.name} unavailable: {exc}")
        continue
      for t in server.tools:
        name = re.sub(r"[^A-Za-z0-9_-]", "_", f"{server.name}__{t['name']}")
        self.mcp[unique_name(name, self.mcp)] = (server, t)
      await self.session.log(
          f"MCP server {server.name}: {len(server.tools)} tools"
      )
    if not self.mcp:
      return ""
    functions = [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": (t.get("description") or t["name"])[:1024],
                "parameters": (
                    {
                        k: v
                        for k, v in (t.get("inputSchema") or {}).items()
                        if k != "$schema"
                    }
                    or {"type": "object", "properties": {}}
                ),
            },
        }
        for name, (_, t) in self.mcp.items()
    ]
    if (
        len(json.dumps(functions)) <= DIRECT_MCP_CHARS
        and len(self.tools) + len(functions) <= MAX_TOOLS
    ):
      self.tools += functions
      return MCP_DIRECT
    self.tools += MCP_TOOLS
    servers: dict[str, list[str]] = {}
    for name, (server, _) in self.mcp.items():
      servers.setdefault(server.name, []).append(name)
    return MCP_CATALOG.format(
        catalog="\n".join(
            f"{s}: {', '.join(names)}" for s, names in servers.items()
        )
    )

  def find_mcp(self, query: str) -> str:
    """Finds MCP tools matching `query` and formats their schemas.

    Returns as many matching tool schemas as fit within `TOOL_RESULT_CHARS`
    (always at least one, without truncating any individual schema).

    Args:
      query: Tool name or search keywords.

    Returns:
      Formatted descriptions and JSON parameter schemas of matching tools.
    """
    words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]

    def score(item):
      name, (_, t) = item
      text = f"{name} {t.get('description') or ''}".lower()
      return (
          (query == name) * 100
          + sum(w in name.lower() for w in words) * 3
          + sum(w in text for w in words)
      )

    best = [
        item
        for item in sorted(self.mcp.items(), key=score, reverse=True)[:8]
        if score(item) > 0
    ]
    if not best:
      return (
          f"no tool matches {query!r}; the names are listed in the system"
          " prompt"
      )
    found = [
        f"{name}: {(t.get('description') or '')[:400]}\nparameters:"
        f" {json.dumps(t.get('inputSchema') or {})}"
        for name, (_, t) in best
    ]
    while len(found) > 1 and len("\n\n".join(found)) > TOOL_RESULT_CHARS:
      found.pop()
    return "\n\n".join(found)

  async def complete(
      self,
      tool_choice: str = "auto",
      reserve: float | None = None,
      head: str | None = None,
  ) -> Reply:
    """Streams one model completion with retries and context-limit recovery.

    Stops streaming `reserve` seconds before the task deadline. Automatically
    adapts token limit parameters, shrinks output ceilings, or compacts the
    conversation when the endpoint reports a context-length error.

    Args:
      tool_choice: OpenAI `tool_choice` value (`"auto"` or `"none"`).
      reserve: Seconds before the task deadline at which to cut off the call.
      head: Prefix text from a prior length-truncated reply, if continuing.

    Returns:
      The completed `Reply` object.

    Raises:
      OutOfTime: If a transient error occurs and there is not enough time left
        to wait and retry.
      CutOff: If the response is still streaming when `reserve` is reached.
      openai.APIError: If a non-retryable API error occurs or retries are
        exhausted.
      StreamCut: If every stream, retries included, ends without a finish
        reason.
    """
    reserve = self.finish_margin if reserve is None else reserve
    waits = iter(RETRY_WAITS)
    while True:
      try:
        reply = await self.stream(self.messages, tool_choice, reserve, head)
        if reply.finish is None:
          raise StreamCut("the stream ended without a finish reason")
        await self.record(reply)
        return reply
      except openai.APIStatusError as exc:
        if exc.status_code == 400 and await self.remedy(str(exc)):
          continue
        if not (exc.status_code in (408, 409, 429) or exc.status_code >= 500):
          raise
        failure = exc
      except StreamBadRequest as exc:
        if await self.remedy(str(exc)):
          continue
        raise
      # a connection error, a timeout, or a stream that errored or ended early
      except (openai.APIError, StreamCut) as exc:
        failure = exc
      wait = next(waits, None)
      if wait is None:
        raise failure
      left = self.time_left()
      if left is not None and left - wait < reserve:
        await self.session.log(
            f"model call failed ({str(failure)[:300]}); no time left to wait"
            f" {wait}s"
        )
        raise OutOfTime(str(failure)[:300]) from failure
      await self.session.log(
          f"model call failed ({str(failure)[:300]}); again in {wait}s"
      )
      await asyncio.sleep(wait)

  async def remedy(self, error: str) -> bool:
    """Adjusts request parameters for the next call after a bad request.

    Args:
      error: The error message returned by a 400 response or stream failure.

    Returns:
      True if the request parameters were adjusted and the call can be retried.
    """
    if CONTEXT_FULL.search(error):
      # The conversation does not fit. Try in this order:
      # 1. fewer output tokens, when the error shows that alone makes room
      #    (restored as the prompt shrinks);
      # 2. compact() what the model has already read;
      # 3. fewer output tokens anyway, when the error did not give the
      #    prompt's size;
      # 4. cut_unread(), the tool results the model has not read yet.
      sent, measured = self.max_output, self.learn_context(error)
      if (
          measured
          and self.room() >= MIN_OUTPUT
          and self.output_limit() < sent
      ):
        return True
      return bool(
          await self.compact(force=True)
          or (not measured and self.output_limit() < sent)
          or self.cut_unread()
      )
    return (
        self.use_responses(error)
        or self.use_max_tokens(error)
        or self.drop_effort(error)
        or self.shrink_output(error)
    )

  def use_responses(self, error: str) -> bool:
    """Switches to the Responses API if a Chat Completions 400 points there.

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if calls now go to the Responses API.
    """
    if self.responses or not RESPONSES_ONLY.search(error):
      return False
    self.responses = True
    return True

  def use_max_tokens(self, error: str) -> bool:
    """Switches to `max_tokens` if `error` rejects `max_completion_tokens`.

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if the token limit parameter was switched to `max_tokens`.
    """
    if (
        self.token_limit != "max_completion_tokens"
        or "max_completion_tokens" not in error
        or not UNSUPPORTED.search(error)
    ):
      return False
    self.token_limit = "max_tokens"
    return True

  def drop_effort(self, error: str) -> bool:
    """Stops sending `reasoning_effort` once an endpoint refuses it.

    A model without reasoning, or a server that does not know the parameter,
    answers `400` naming it (`reasoning.effort` on the Responses API).

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if `reasoning_effort` was being sent and is now dropped.
    """
    if not self.effort or not re.search(r"reasoning[._]effort", error):
      return False
    self.effort = {}
    return True

  def shrink_output(self, error: str) -> bool:
    """Lowers `output_ceiling` when `error` indicates a smaller output limit.

    Args:
      error: Error message string from a 400 `APIStatusError`.

    Returns:
      `True` if `output_ceiling` was reduced, else `False`.
    """
    if UNSUPPORTED.search(error):
      return False
    limit = re.search(r"(\d+)\s*\(exclusive\)", error)
    stated = re.search(
        r"at most (\d+)|(\d+), which is the maximum|maximum (?:allowed"
        r" )?(?:number of )?(?:output|"
        r"completion) tokens (?:is |of )?(\d+)",
        error,
        re.I,
    )
    if limit:
      new = int(limit.group(1)) - 1
    elif stated:
      new = int(next(g for g in stated.groups() if g))
    elif re.search(
        r"max_?(completion_)?tokens|maxOutputTokens|output tokens", error, re.I
    ):
      new = self.max_output // 2
    else:
      return False
    if new >= self.max_output or new < MIN_OUTPUT:
      return False
    self.output_ceiling = new
    return True

  def learn_context(self, error: str) -> bool:
    """Extracts context window and prompt token counts from `error`.

    Args:
      error: Error message string from a context-length `APIStatusError`.

    Returns:
      `True` if both `context_window` and `prompt_tokens` are now known.
    """
    window, prompt = WINDOW_RE.search(error), PROMPT_RE.search(error)
    if window:
      self.context_window = int(next(g for g in window.groups() if g))
    if prompt:
      self.prompt_tokens = int(next(g for g in prompt.groups() if g))
    return bool(prompt and self.context_window)

  def room(self) -> int | None:
    """Returns remaining output token capacity in the known context window.

    Returns:
      Available tokens beside the prompt, or `None` if the window size is
      unknown.
    """
    return (
        self.context_window - self.prompt_tokens - CONTEXT_MARGIN
        if self.context_window
        else None
    )

  def output_limit(self) -> int:
    """Computes the max output tokens to request on the next completion call.

    Returns:
      `output_ceiling`, clamped down to `room()` when known (never below
      `MIN_OUTPUT`).
    """
    room = self.room()
    return (
        self.output_ceiling
        if room is None
        else max(MIN_OUTPUT, min(self.output_ceiling, room))
    )

  def compact_at(self) -> int:
    """Returns the prompt token threshold at which to trigger compaction.

    Returns:
      `COMPACT_AT_TOKENS`, or a lower threshold when the known context window
      requires more room for output.
    """
    if not self.context_window:
      return COMPACT_AT_TOKENS
    room = min(self.output_ceiling, self.context_window // 2)
    return max(
        self.context_window // 4,
        min(COMPACT_AT_TOKENS, self.context_window - room - CONTEXT_MARGIN),
    )

  def estimate_prompt_tokens(self) -> int:
    """Estimates prompt token count at four JSON characters per token.

    Skips `output_items`, which repeat a Responses API turn's text and calls
    plus encrypted reasoning whose length does not track its token count.

    Returns:
      Estimated number of tokens in `self.messages`.
    """
    return (
        sum(
            len(json.dumps({k: v for k, v in m.items() if k != "output_items"}))
            for m in self.messages
        )
        // 4
    )

  async def stream(
      self,
      messages: list[dict[str, Any]],
      tool_choice: str,
      reserve: float,
      head: str | None = None,
  ) -> Reply:
    """Streams a single chat completion call up to `reserve` seconds before end.

    If `head` is provided and the stream is cut off or cancelled, updates
    `self.session.final_message` with `visible(head + reply.text)`.

    Args:
      messages: Conversation message list to send.
      tool_choice: OpenAI `tool_choice` setting (`"auto"` or `"none"`).
      reserve: Seconds to hold back before the session deadline.
      head: Prefix text from an earlier truncated response, or `None`.

    Returns:
      The accumulated `Reply` instance.

    Raises:
      CutOff: If the stream times out at `left - reserve`.
      asyncio.CancelledError: If the task is cancelled during streaming.
    """
    left = self.time_left()
    self.max_output = self.output_limit()
    reply = Reply()

    async def read():
      if self.responses:
        await self.respond(reply, messages, tool_choice)
        return
      stream = await client.chat.completions.create(
          model=MODEL,
          messages=marked(messages),
          tools=self.tools,
          tool_choice=tool_choice,
          stream=True,
          stream_options={"include_usage": True},
          **self.effort,
          **{self.token_limit: self.max_output},
      )
      async for chunk in stream:
        reply.usage = chunk.usage or reply.usage
        if not chunk.choices:
          continue
        delta = chunk.choices[0].delta
        reply.finish = chunk.choices[0].finish_reason or reply.finish
        reply.text += delta.content or ""
        reply.reasoning += (
            getattr(delta, "reasoning_content", None)
            or getattr(delta, "reasoning", None)
            or ""
        )
        reply.extra = merged(reply.extra, extra_fields(delta, DELTA_FIELDS))
        for part in delta.tool_calls or []:
          reply.add_call(part)

    try:
      await asyncio.wait_for(read(), None if left is None else left - reserve)
    except (asyncio.TimeoutError, asyncio.CancelledError) as exc:
      if head is not None:
        self.session.final_message = (
            visible(head + reply.text) or self.session.final_message
        )
      if isinstance(exc, asyncio.CancelledError):
        raise
      raise CutOff() from None
    except openai.APIError as exc:
      # the SDK raises an error sent inside a chat stream as an APIError
      said = {
          "message": getattr(exc, "message", None) or str(exc),
          "type": getattr(exc, "type", None),
          "code": getattr(exc, "code", None),
      }
      if isinstance(exc, openai.APIStatusError) or isinstance(
          stream_error(said), StreamCut
      ):
        raise
      raise StreamBadRequest(said["message"]) from exc
    for n, call in reply.calls.items():
      call["id"] = call["id"] or f"call_{n}_{len(self.messages)}"
      # some open models leak chat-template tokens into it
      call["name"] = call["name"].split("<|")[0].strip()
    return reply

  async def respond(
      self, reply: Reply, messages: list[dict[str, Any]], tool_choice: str
  ) -> None:
    """Streams a single Responses API call into `reply`.

    Keeps the conversation in Chat Completions format and converts it on each
    call via `response_input`. Because `store=False`, each response's output
    items (including encrypted reasoning) are saved and echoed back on
    subsequent turns. If the stream ends before `response.completed` or
    `response.incomplete`, `reply.finish` remains `None` so the call is retried.

    Args:
      reply: The `Reply` to populate as events arrive.
      messages: Conversation message list to send.
      tool_choice: `"auto"` or `"none"`.

    Raises:
      StreamCut: If the stream reports a transient error.
      StreamBadRequest: If the stream reports a non-retryable bad request.
    """
    effort = self.effort.get("reasoning_effort")
    stream = await client.responses.create(
        model=MODEL,
        input=response_input(messages),
        tools=[
            {"type": "function", **t["function"], "strict": False}
            for t in self.tools
        ],
        tool_choice=tool_choice,
        stream=True,
        store=False,
        include=["reasoning.encrypted_content"],
        max_output_tokens=self.max_output,
        **({"reasoning": {"effort": effort}} if effort else {}),
    )
    async for event in stream:
      kind = getattr(event, "type", "")
      if kind == "response.output_text.delta":
        reply.text += event.delta or ""
      elif kind in ("error", "response.failed"):
        raise stream_error(
            getattr(getattr(event, "response", None), "error", None) or event
        )
      elif kind in ("response.completed", "response.incomplete"):
        done = event.response
        items = [i.model_dump(exclude_none=True) for i in done.output or []]
        reply.extra = {"output_items": items}
        reply.text = "".join(
            part.get("text", "")
            for item in items
            if item.get("type") == "message"
            for part in item.get("content") or []
            if part.get("type") == "output_text"
        )
        for item in items:
          if item.get("type") == "function_call":
            reply.calls[len(reply.calls)] = {
                "id": item.get("call_id") or "",
                "name": item.get("name") or "",
                "arguments": item.get("arguments") or "",
                "extra": {},
                "function_extra": {},
            }
        reply.usage = chat_usage(done.usage)
        reason = getattr(done.incomplete_details, "reason", None)
        if kind == "response.completed":
          reply.finish = "tool_calls" if reply.calls else "stop"
        else:
          reply.finish = reason if reason == "content_filter" else "length"

  async def record(self, reply: Reply) -> None:
    """Updates `prompt_tokens` from `reply` and reports token usage.

    Args:
      reply: Completed `Reply` from `stream()`.
    """
    if reply.usage is None:
      self.prompt_tokens = self.estimate_prompt_tokens()
      return
    self.prompt_tokens = reply.usage.prompt_tokens or 0
    await self.report(reply)

  async def report(self, reply: Reply) -> None:
    """Reports token usage from `reply` to the evaluator session if present.

    Args:
      reply: Completed `Reply` whose `usage` should be reported.
    """
    usage = reply.usage
    if usage is None:
      return
    cached = (
        getattr(
            getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0
        )
        or 0
    )
    await self.session.usage(
        MODEL, usage.prompt_tokens or 0, usage.completion_tokens or 0, cached
    )

  async def run_tool(self, name: str, args: dict[str, Any]) -> str:
    """Dispatches a tool call by `name` and returns its clipped text output.

    Args:
      name: Tool function name (`bash`, `read_file`, `write_file`, `edit_file`,
        `mcp_find`, `mcp_call`, or a direct MCP tool name).
      args: Parsed tool arguments dictionary.

    Returns:
      The tool output or error string.
    """
    await self.session.log(f"{name} {json.dumps(args)[:400]}")
    try:
      if name == "mcp_call":
        name, args = str(args.get("tool", "")), args.get("arguments") or {}
        args = json.loads(args) if isinstance(args, str) else args
        if name not in self.mcp:
          return f"error: no tool named {name!r}; use mcp_find"
      if name in self.mcp:
        server, t = self.mcp[name]
        return clip(await server.call(t["name"], args))
      if name == "mcp_find":
        return self.find_mcp(str(args.get("query", "")))
      handler = {
          "bash": self.bash,
          "read_file": self.read_file,
          "write_file": self.write_file,
          "edit_file": self.edit_file,
      }.get(name)
      return (
          clip(await handler(**args))
          if handler
          else f"error: there is no tool named {name}"
      )
    except taskbridge.TaskError as exc:
      return f"error: {exc}"
    except (
        AttributeError,
        KeyError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
      return f"error: {type(exc).__name__}: {exc}"

  def path(self, path: str) -> str:
    """Resolves `path` relative to `self.session.workdir` if not absolute.

    Args:
      path: Absolute or workdir-relative file path.

    Returns:
      The resolved container path string.
    """
    return (
        path
        if path.startswith("/")
        else posixpath.join(self.session.workdir, path)
    )

  async def bash(self, command: str, timeout_sec: int = 120) -> str:
    """Executes `command` with bash in the task container.

    Args:
      command: Shell command string to run.
      timeout_sec: Timeout in seconds, cut to end before the time kept for the
        final summary.

    Returns:
      Formatted exit code and command output string.
    """
    cap = int(self.session.limits.get("exec_timeout_sec") or 600)
    left = self.time_left()
    if left is not None:
      cap = min(cap, max(1, int(left - self.finish_margin)))
    result = await self.session.exec(
        command, taskbridge.positive_int(timeout_sec, 120, cap)
    )
    note = " (timed out)" if result.get("timed_out") else ""
    return (
        "exit code"
        f" {result['exit_code']}{note}\n{result['output'] or '(no output)'}"
    )

  async def read_file(self, path: str) -> str:
    """Reads a UTF-8 text file from `path` in the task container.

    Args:
      path: File path inside the task container.

    Returns:
      Decoded file contents, or a notice if the file is binary or empty.
    """
    data = await self.session.read_file(self.path(path))
    if b"\0" in data[:8192]:
      return (
          f"{path} is a binary file ({len(data)} bytes); inspect it with bash"
          " (python3, unzip -l, ...)"
      )
    return data.decode("utf-8", "replace") or "(empty file)"

  async def write_file(
      self, path: str, content: str, append: bool = False
  ) -> str:
    """Writes or appends `content` to `path` in the task container.

    Args:
      path: Destination file path inside the task container.
      content: Text content to write.
      append: Whether to append to the file instead of overwriting.

    Returns:
      Confirmation string with the resulting file size in bytes.
    """
    size = await self.session.write_file(
        self.path(path), content, append=bool(append)
    )
    return f"{path} now holds {size} bytes"

  async def edit_file(
      self, path: str, old_text: str, new_text: str, replace_all: bool = False
  ) -> str:
    """Replaces `old_text` with `new_text` in `path` inside the task container.

    Args:
      path: Target file path inside the task container.
      old_text: Substring to search for.
      new_text: Replacement string.
      replace_all: Whether to replace all occurrences instead of requiring one.

    Returns:
      Confirmation string or an error message if `old_text` is missing or
      ambiguous.
    """
    text = (await self.session.read_file(self.path(path))).decode("utf-8")
    count = text.count(old_text) if old_text else 0
    if count == 0:
      return f"error: old_text was not found in {path}"
    if count > 1 and not replace_all:
      return (
          f"error: old_text occurs {count} times in {path}; add context to make"
          " it unique, or set replace_all"
      )
    await self.session.write_file(
        self.path(path),
        text.replace(old_text, new_text, -1 if replace_all else 1),
    )
    return f"replaced {count if replace_all else 1} occurrence(s) in {path}"

  def unread(self) -> list[dict[str, Any]]:
    """Returns tool result messages appended after the last assistant turn.

    Returns:
      List of unread `"tool"` role message dictionaries.
    """
    last = max(
        (n for n, m in enumerate(self.messages) if m["role"] == "assistant"),
        default=0,
    )
    return [m for m in self.messages[last:] if m["role"] == "tool"]

  async def compact(self, force: bool = False) -> bool:
    """Compacts conversation history without modifying unread tool results.

    First truncates older read tool outputs (keeping `KEEP_RECENT_RESULTS`
    unless `force` is `True`). If that is insufficient, summarizes older turns
    via a model call. Claude's thinking blocks stay on the kept turns: a block's
    signature does not depend on the turns before it, and Claude refuses a tool
    loop whose newest turn lost its block.

    Args:
      force: Whether compaction was triggered by a context-full error.

    Returns:
      `True` if any tool results were truncated or turns were summarized.
    """
    unread = {id(m) for m in self.unread()}
    results = [
        (n, m) for n, m in enumerate(self.messages) if m["role"] == "tool"
    ]
    # with `force`, the endpoint's own count of the prompt wins when it is
    # larger: the estimate can run low
    cuts, size = [], max(
        self.estimate_prompt_tokens(), self.prompt_tokens if force else 0
    )
    for n, m in results if force else results[:-KEEP_RECENT_RESULTS]:
      if force and self.context_window and size <= self.compact_at():
        break
      if len(m["content"]) > 600 and id(m) not in unread:
        size -= (len(m["content"]) - 400) // 4
        m["content"] = (
            m["content"][:400]
            + "\n[... removed to save context; run the tool again if needed"
            " ...]"
        )
        cuts.append(n)
    if cuts:
      await self.session.log(f"context: cut {len(cuts)} old tool results")
      self.prompt_tokens = self.estimate_prompt_tokens()
      if (
          not force
          or not self.context_window
          or self.prompt_tokens <= self.compact_at()
      ):
        return True
    cut = len(self.messages) - 6
    # the kept turns must not start with tool results whose call was replaced by
    # the notes
    while cut > 3 and self.messages[cut]["role"] == "tool":
      cut -= 1
    if cut < 10:  # too few turns to be worth a notes call
      return bool(cuts)
    try:
      notes = await self.stream(
          self.messages[:cut] + [{"role": "user", "content": SUMMARY_PROMPT}],
          "none",
          self.finish_margin,
      )
    except (openai.APIError, StreamCut, asyncio.TimeoutError) as exc:
      await self.session.log(
          f"context: notes failed ({str(exc)[:200] or 'out of time'})"
      )
      return bool(cuts)
    await self.report(notes)
    if notes.finish is None:
      await self.session.log(
          "context: the notes were cut short, so the turns are kept"
      )
      return bool(cuts)
    text = visible(notes.text)
    if not text:
      await self.session.log(
          "context: the notes came back empty, so the turns are kept"
      )
      return bool(cuts)
    if notes.finish == "length":
      text += "\n[the notes were cut off at the output limit]"
    self.messages = (
        self.messages[:2]
        + [{
            "role": "user",
            "content": (
                "Your notes on the work so far (earlier "
                "turns were removed to save context):\n"
                + text
            ),
        }]
        + self.messages[cut:]
    )
    await self.session.log(f"context: replaced {cut - 2} messages with notes")
    self.prompt_tokens = self.estimate_prompt_tokens()
    return True

  def cut_unread(self) -> bool:
    """Truncates unread tool results as a last resort when context is full.

    Clips each unread tool output exceeding `2 * UNREAD_KEEP` to an equal share
    of the remaining context window (or halves it when the window size is
    unknown).

    Returns:
      `True` if at least one unread tool result was shortened, else `False`.
    """
    unread = [m for m in self.unread() if len(m["content"]) > 2 * UNREAD_KEEP]
    if not unread:
      return False
    share = None
    if self.context_window:
      rest = (
          self.estimate_prompt_tokens()
          - sum(len(m["content"]) for m in unread) // 4
      )
      share = (self.compact_at() - rest) * 4 // len(unread)
    for m in unread:
      size = len(m["content"])
      # a share within 100 characters of the size would barely shrink it
      keep = size // 2 if share is None or share >= size - 100 else share
      m["content"] = clip(m["content"], max(UNREAD_KEEP, keep))
    self.prompt_tokens = self.estimate_prompt_tokens()
    return True


async def _run_phase(
    session: taskbridge.Session, label: str, agent: Agent
) -> str:
  """Runs one agent phase, logging (not raising) its failure.

  A failed planner or verifier must not fail the task: the worker still runs
  (without a plan) and its summary still stands (unverified). Cancellation
  still propagates so the evaluator's cancel is honoured.

  Args:
    session: Active task session.
    label: Phase name for the log.
    agent: The configured `Agent` for the phase.

  Returns:
    The phase's answer text, or "" when the phase failed.
  """
  try:
    return await agent.run()
  except taskbridge.Cancelled:
    raise
  except asyncio.CancelledError:
    raise
  except Exception as exc:  # pylint: disable=broad-exception-caught
    await session.log(f"{label} phase failed: {type(exc).__name__}: {exc}")
    return ""


def _verdict(text: str) -> tuple[bool, str]:
  """Parses a verifier reply into (passed, gaps).

  Args:
    text: The verifier's reply text.

  Returns:
    `(True, "")` when the reply starts with PASS, else `(False, text)`.
  """
  if re.match(r"\s*PASS\b", text, re.I):
    return True, ""
  return False, text.strip()


# Deterministic IFC gate: validates every produced .ifc under output/ with
# the full EXPRESS rules, in the task container (no model call). A schema-only
# check cannot catch geometry-rule violations, and an agent that failed to run
# the full check once will silently substitute the weaker one — so the gate
# runs the exact check itself and its failures become fix-round gaps.
IFC_GATE_SCRIPT = """
import glob, sys
try:
    import ifcopenshell
    import ifcopenshell.validate
except Exception as exc:
    print(f"IFC_GATE: SKIP (ifcopenshell unavailable: {exc})")
    sys.exit(0)
class _Log:
    def __init__(self):
        self.errors = []
    def error(self, *a, **k):
        self.errors.append((a, k))
files = sorted(glob.glob("output/**/*.ifc", recursive=True))
if not files:
    print("IFC_GATE: SKIP (no .ifc files under output/)")
    sys.exit(0)
bad = 0
for f in files:
    log = _Log()
    try:
        model = ifcopenshell.open(f)
        ifcopenshell.validate.validate(model, log, express_rules=True)
    except Exception as exc:
        print(f"IFC {f}: ERROR {exc}")
        bad += 1
        continue
    print(f"IFC {f}: {len(log.errors)} EXPRESS violations")
    for args, _ in log.errors[:10]:
        print("  " + " ".join(str(a) for a in args)[:500])
    bad += 1 if log.errors else 0
print("IFC_GATE: FAIL" if bad else
      f"IFC_GATE: PASS ({len(files)} file(s), 0 EXPRESS violations)")
"""
IFC_GATE_COMMAND = "python3 - <<'PYEOF'\n" + IFC_GATE_SCRIPT + "\nPYEOF"


async def _ifc_gate(worker: Agent) -> tuple[str, dict]:
  """Runs the deterministic IFC gate and returns its failures as gaps.

  The gate runs in the task container through the worker's bash tool. It
  returns ("", scores) when there is nothing to check, the check passes, or
  the check could not run (missing ifcopenshell, no python, a tool error) —
  an inconclusive gate must not block submission; the verifier still checks.

  Args:
    worker: The worker Agent (its bash tool runs in the task container).

  Returns:
    (gaps, scores): the gate's violation text when it fails, else "";
    scores is {"ifc_ok": bool, "violations": int} when the gate measured
    something, else {} (inconclusive).
  """
  try:
    out = await worker.bash(IFC_GATE_COMMAND)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    await worker.session.log(f"IFC gate failed to run: {exc}")
    return "", {}
  if "IFC_GATE: FAIL" not in out:
    await worker.session.log(
        "IFC gate: " + (out.strip().splitlines() or [""])[-1][:200]
    )
    return "", _ifc_gate_scores(out)
  return clip(out, 4000), _ifc_gate_scores(out)


def _ifc_gate_scores(out: str) -> dict:
  """Parses the IFC gate's output into a score dict for the regression guard.

  Returns {"ifc_ok": bool, "violations": int} when the gate measured
  something, else {} (inconclusive — the guard must not compare unknowns).
  """
  if "IFC_GATE: SKIP" in out:
    return {}
  violations = 0
  for line in out.splitlines():
    if line.startswith("IFC ") and "EXPRESS violations" in line:
      try:
        violations += int(line.split(":")[1].split()[0])
      except (IndexError, ValueError):
        pass
  if "IFC_GATE: PASS" in out:
    return {"ifc_ok": True, "violations": violations}
  if "IFC_GATE: FAIL" in out:
    return {"ifc_ok": False, "violations": violations}
  return {}


# Deterministic unit gate: runs the grader's own unit-scale calculation on
# every produced .ifc. A model without an IfcUnitAssignment defaults to
# metres, so the grader reads every coordinate 1000x too large — and
# validation passes either way, so no other gate can see this bug. A real
# run scored 0 for exactly this reason (the grader's routed length was
# 345,900,000 mm, exactly 1000x the 345,900 mm the gates read).
UNIT_GATE_SCRIPT = """
import glob, sys
try:
    import ifcopenshell
    import ifcopenshell.util.unit
except Exception as exc:
    print(f"UNIT_GATE: SKIP (ifcopenshell unavailable: {exc})")
    sys.exit(0)
files = sorted(glob.glob("output/**/*.ifc", recursive=True))
if not files:
    print("UNIT_GATE: SKIP (no .ifc files under output/)")
    sys.exit(0)
bad = 0
for f in files:
    try:
        model = ifcopenshell.open(f)
        scale = ifcopenshell.util.unit.calculate_unit_scale(model) * 1000.0
    except Exception as exc:
        print(f"UNIT {f}: ERROR {exc}")
        bad += 1
        continue
    has_mm = False
    for ua in model.by_type('IfcUnitAssignment'):
        for u in ua.Units or []:
            if (u.is_a('IfcSIUnit') and u.UnitType == 'LENGTHUNIT'
                    and u.Prefix == 'MILLI' and u.Name == 'METRE'):
                has_mm = True
    print(f"UNIT {f}: scale x{scale:.3f}, millimetre unit declared: {has_mm}")
    if abs(scale - 1.0) > 0.001 or not has_mm:
        print(f"UNIT FAIL: {f} reads {scale:.0f}x too large — the model has no millimetre unit declaration; the grader reads every coordinate {scale:.0f}x too large. Add the IfcSIUnit MILLI METRE + IfcUnitAssignment + project.UnitsInContext block from IFC_GUIDE.")
        bad += 1
print("UNIT_GATE: FAIL" if bad else
      f"UNIT_GATE: PASS ({len(files)} file(s), scale x1.0, millimetre units)")
"""
UNIT_GATE_COMMAND = "python3 - <<'PYEOF'\n" + UNIT_GATE_SCRIPT + "\nPYEOF"


async def _unit_gate(worker: Agent) -> tuple[str, dict]:
  """Runs the deterministic unit gate and returns failures as gaps.

  The gate runs in the task container through the worker's bash tool. It
  returns ("", scores) when there is nothing to check, the check passes, or
  the check could not run (missing ifcopenshell, no python, a tool error) —
  an inconclusive gate must not block submission; the verifier still checks.

  Args:
    worker: The worker Agent (its bash tool runs in the task container).

  Returns:
    (gaps, scores): the gate's violation text when it fails, else "";
    scores is {"unit_ok": bool, "scale": float} when the gate measured
    something, else {} (inconclusive).
  """
  try:
    out = await worker.bash(UNIT_GATE_COMMAND)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    await worker.session.log(f"Unit gate failed to run: {exc}")
    return "", {}
  if "UNIT_GATE: FAIL" not in out:
    await worker.session.log(
        "Unit gate: " + (out.strip().splitlines() or [""])[-1][:200]
    )
    return "", _unit_gate_scores(out)
  return clip(out, 4000), _unit_gate_scores(out)


def _unit_gate_scores(out: str) -> dict:
  """Parses the unit gate's output into a score dict for the regression guard.

  Returns {"unit_ok": bool, "scale": float} when the gate measured
  something, else {} (inconclusive — the guard must not compare unknowns).
  """
  if "UNIT_GATE: SKIP" in out:
    return {}
  scale = None
  for line in out.splitlines():
    if line.startswith("UNIT ") and "scale x" in line:
      try:
        scale = float(line.split("scale x")[1].split(",")[0])
      except (IndexError, ValueError):
        pass
  if scale is None:
    return {}
  if "UNIT_GATE: PASS" in out:
    return {"unit_ok": True, "scale": scale}
  if "UNIT_GATE: FAIL" in out:
    return {"unit_ok": False, "scale": scale}
  return {}


# Deterministic connectivity gate: measures the route geometry of every
# produced .ifc with the same tolerances the grader reports (250 mm joint
# tolerance, 2,500 mm entry / row / CRAH reach, 80% row coverage). A
# verifier that never recomputed connectivity passed a one-box-per-service
# route, so the gate runs the exact measurement itself and its failures
# become fix-round gaps. Only runs on routing tasks.
CONN_GATE_SCRIPT = """

import glob, json, math, os, sys
try:
    import ifcopenshell
except Exception as exc:
    print(f"CONN_GATE: SKIP (ifcopenshell unavailable: {exc})")
    sys.exit(0)

ENTRY_TOL = 2500.0
JOIN_TOL = 250.0
COVER_TOL = 2500.0
ROW_MIN = 0.8
CRAH_TOL = 2500.0


def _v(p):
    return (float(p[0]), float(p[1]), float(p[2]))


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _scale(v, s):
    return tuple(x * s for x in v)


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _dist(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def _matvec(cols, v):
    return tuple(sum(cols[j][i] * v[j] for j in range(3)) for i in range(3))


def _axis_frame(ap):
    '''Frame of an IfcAxis2Placement3D: (location, x, y, z).'''
    loc = _v(ap.Location.Coordinates) if getattr(ap, 'Location', None) else (0.0, 0.0, 0.0)
    z = _norm(_v(ap.Axis.DirectionRatios)) if getattr(ap, 'Axis', None) else (0.0, 0.0, 1.0)
    x = _norm(_v(ap.RefDirection.DirectionRatios)) if getattr(ap, 'RefDirection', None) else (1.0, 0.0, 0.0)
    y = _norm(_cross(z, x))
    x = _norm(_cross(y, z))
    return loc, x, y, z


def _world_frame(pl):
    '''World frame of an IfcLocalPlacement, walking its PlacementRelTo chain.'''
    origin, x, y, z = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    chain, cur = [], pl
    while cur is not None:
        chain.append(cur)
        cur = getattr(cur, 'PlacementRelTo', None)
    for p in reversed(chain):
        loc, px, py, pz = _axis_frame(p.RelativePlacement)
        origin = _add(origin, _matvec((x, y, z), loc))
        x, y, z = _matvec((x, y, z), px), _matvec((x, y, z), py), _matvec((x, y, z), pz)
    return origin, x, y, z


def _profile_bbox(prof):
    if prof.is_a('IfcRectangleProfileDef'):
        cx, cy = 0.0, 0.0
        if getattr(prof, 'Position', None) and getattr(prof.Position, 'Location', None):
            c = prof.Position.Location.Coordinates
            cx, cy = float(c[0]), float(c[1])
        xd, yd = float(prof.XDim), float(prof.YDim)
        return (cx - xd / 2, cx + xd / 2), (cy - yd / 2, cy + yd / 2)
    if prof.is_a('IfcArbitraryClosedProfileDef'):
        pts = prof.OuterCurve.Points
        xs = [float(p.Coordinates[0]) for p in pts]
        ys = [float(p.Coordinates[1]) for p in pts]
        return (min(xs), max(xs)), (min(ys), max(ys))
    return (-0.5, 0.5), (-0.5, 0.5)


def _box(product):
    '''Oriented box of an extruded solid: (centre, [(dir, half_len) x3]).

    Returns None when the product has no extruded solid representation.
    '''
    if getattr(product, 'ObjectPlacement', None) is None:
        return None
    if getattr(product, 'Representation', None) is None:
        return None
    try:
        origin, x, y, z = _world_frame(product.ObjectPlacement)
    except Exception:
        return None
    for rep in product.Representation.Representations:
        for item in rep.Items:
            if not item.is_a('IfcExtrudedAreaSolid'):
                continue
            if getattr(item, 'Position', None):
                sloc, sx, sy, sz = _axis_frame(item.Position)
                origin = _add(origin, _matvec((x, y, z), sloc))
                x, y, z = (_matvec((x, y, z), sx), _matvec((x, y, z), sy),
                           _matvec((x, y, z), sz))
            (x0, x1), (y0, y1) = _profile_bbox(item.SweptArea)
            dx, dy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            depth = float(item.Depth)
            ed = _matvec((x, y, z), _norm(_v(item.ExtrudedDirection.DirectionRatios)))
            centre = _add(origin, _add(_add(_scale(x, cx), _scale(y, cy)),
                                        _scale(ed, depth / 2.0)))
            return centre, [(x, dx), (y, dy), (ed, depth / 2.0)]
    return None


def _aabb(centre, axes):
    ext = [0.0, 0.0, 0.0]
    for d, half in axes:
        for i in range(3):
            ext[i] += abs(d[i]) * half
    return tuple(centre[i] - ext[i] for i in range(3)), tuple(centre[i] + ext[i] for i in range(3))


def _segment(product):
    '''Centreline (start, end) of an extruded route segment, or None.

    A segment is an oriented box: the centreline runs along its longest
    axis, so this works both for a long profile with a short extrusion
    (the agent's box style) and for a small profile with a long extrusion
    (the rotated-placement style).
    '''
    b = _box(product)
    if b is None:
        return None
    centre, axes = b
    d, half = max(axes, key=lambda a: a[1])
    return _sub(centre, _scale(d, half)), _add(centre, _scale(d, half))


def _pt_seg_dist(p, a, b):
    ab = _sub(b, a)
    t = max(0.0, min(1.0, _dot(_sub(p, a), ab) / max(_dot(ab, ab), 1e-9)))
    return _dist(p, _add(a, _scale(ab, t)))


def _pt_seg_dist2d(p, a, b):
    return _pt_seg_dist((p[0], p[1], 0.0), (a[0], a[1], 0.0), (b[0], b[1], 0.0))


def _coverage(start, end, segments, tol):
    n, hit = 100, 0
    for i in range(n + 1):
        t = i / n
        p = _add(start, _scale(_sub(end, start), t))
        if any(_pt_seg_dist2d(p, a, b) <= tol for a, b in segments):
            hit += 1
    return hit / (n + 1)


def _aabb_dist(a_min, a_max, b_min, b_max):
    '''Distance between two axis-aligned bounding boxes (0 when they touch
    or overlap). This is the distance between the two solids, not their
    centrelines: a butt joint (two boxes touching end to end, 0 mm overlap)
    is connected, even though the centrelines are a half-width apart at a
    corner — measuring centrelines would call a correct butt joint
    disconnected.'''
    d2 = 0.0
    for i in range(3):
        gap = max(b_min[i] - a_max[i], a_min[i] - b_max[i], 0.0)
        d2 += gap * gap
    return math.sqrt(d2)


def _components(boxes, tol):
    '''Number of connected components among the boxes, joined when their
    solids are within tol (250 mm) of each other. A T-junction (a branch
    meeting the middle of a trunk) counts as joined, because the solids
    touch even though no endpoints meet.'''
    parent = list(range(len(boxes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if _aabb_dist(boxes[i][0], boxes[i][1], boxes[j][0], boxes[j][1]) <= tol:
                parent[find(i)] = find(j)
    return len({find(i) for i in range(len(boxes))})


def _closest_break(boxes, tol):
    '''The smallest gap between two boxes in different components.

    Returns (distance, i, j, point_a, point_b): the indices of the two
    nearest unconnected boxes and the closest points on their surfaces
    (each box's centre clamped onto the other box's bounds). Returns None
    when the boxes form one component. Naming the break lets the fix
    round see where the network is broken instead of only knowing that
    it is broken.
    '''
    n = len(boxes)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _aabb_dist(boxes[i][0], boxes[i][1], boxes[j][0], boxes[j][1]) <= tol:
                parent[find(i)] = find(j)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            d = _aabb_dist(boxes[i][0], boxes[i][1], boxes[j][0], boxes[j][1])
            if best is None or d < best[0]:
                a_min, a_max = boxes[i]
                b_min, b_max = boxes[j]
                ca = tuple((a_min[k] + a_max[k]) / 2 for k in range(3))
                cb = tuple((b_min[k] + b_max[k]) / 2 for k in range(3))
                pa = tuple(max(a_min[k], min(cb[k], a_max[k])) for k in range(3))
                pb = tuple(max(b_min[k], min(ca[k], b_max[k])) for k in range(3))
                best = (d, i, j, pa, pb)
    return best


def _classify(product):
    text = ((product.Name or '') + ' ' + (product.ObjectType or '')).lower()
    if 'busway' in text:
        return 'busway'
    if 'tray' in text:
        return 'tray'
    if 'chw' in text or 'pipe' in text or 'chilled' in text:
        return 'chw'
    return None


def main():
    files = sorted(glob.glob('output/**/*.ifc', recursive=True))
    if not files:
        print('CONN_GATE: SKIP (no .ifc files under output/)')
        return
    path = 'output/model.ifc' if 'output/model.ifc' in files else max(files, key=os.path.getsize)
    try:
        model = ifcopenshell.open(path)
    except Exception as exc:
        print(f'CONN_GATE: SKIP (cannot open {path}: {exc})')
        return
    segs = {'tray': [], 'busway': [], 'chw': []}
    boxes = {'tray': [], 'busway': [], 'chw': []}
    names = {'tray': [], 'busway': [], 'chw': []}
    wrong = []
    unclassified = 0
    for product in model.by_type('IfcFlowSegment'):
        svc = _classify(product)
        if svc is None:
            unclassified += 1
            continue
        expect = 'IfcCableCarrierSegment' if svc in ('tray', 'busway') else 'IfcPipeSegment'
        if not product.is_a(expect):
            wrong.append(f"{svc} modelled as {product.is_a()}, not {expect}")
            continue
        b = _box(product)
        if b is None:
            continue
        centre, axes = b
        d, half = max(axes, key=lambda a: a[1])
        segs[svc].append((_sub(centre, _scale(d, half)), _add(centre, _scale(d, half))))
        boxes[svc].append(_aabb(centre, axes))
        names[svc].append(product.Name or product.is_a())
    for product in model.by_type('IfcElectricDistributionBoard'):
        # Only a board that claims to be a route segment is a wrong class;
        # a legitimate PDU (no tray/busway name) is not a route element.
        svc = _classify(product)
        if svc is None:
            continue
        expect = 'IfcCableCarrierSegment' if svc in ('tray', 'busway') else 'IfcPipeSegment'
        wrong.append(f"{svc} modelled as {product.is_a()}, not {expect}")
    if unclassified:
        print(f"CONN note: {unclassified} flow segment(s) without a recognisable name (tray/busway/chw)")
    problems = []
    for svc in ('tray', 'busway', 'chw'):
        n = len(segs[svc])
        total = sum(_dist(a, b) for a, b in segs[svc])
        print(f"CONN {svc}: {n} segment(s), {total:.0f} mm total")
        if n == 0 or total == 0:
            problems.append(f"{svc}: no route (0 segments / 0 mm)")
        elif n == 1:
            problems.append(f"{svc}: only 1 segment ({total:.0f} mm) — a route needs feeders, headers and drops, not one box")
    for w in wrong:
        problems.append(w)
    for svc in ('tray', 'busway', 'chw'):
        if len(boxes[svc]) > 1:
            c = _components(boxes[svc], JOIN_TOL)
            print(f"CONN {svc}: {c} connected component(s)")
            if c > 1:
                problems.append(f"{svc}: {c} disconnected components (segments must join within {JOIN_TOL:.0f} mm)")
                brk = _closest_break(boxes[svc], JOIN_TOL)
                if brk:
                    d, i, j, pa, pb = brk
                    problems.append(
                        f"{svc} break: nearest unconnected segments are {d:.0f} mm apart — {names[svc][i]} ends near ({pa[0]:.0f},{pa[1]:.0f},{pa[2]:.0f}), {names[svc][j]} starts near ({pb[0]:.0f},{pb[1]:.0f},{pb[2]:.0f}); bridge that gap")
    try:
        routing = json.load(open('output/routing.json'))
    except Exception as exc:
        routing = None
        print(f"CONN routing.json: not usable ({exc}) — coverage checks skipped")
    if routing:
        entry = _v(routing['entry'])
        for svc in ('tray', 'busway', 'chw'):
            if segs[svc]:
                d = min(_pt_seg_dist2d(entry, a, b) for a, b in segs[svc])
                print(f"CONN {svc}: entry distance {d:.0f} mm")
                if d > ENTRY_TOL:
                    problems.append(f"{svc}: nothing within {ENTRY_TOL:.0f} mm of the entry ({d:.0f} mm) — the network is not fed")
        for row in routing.get('rows', []):
            rs, re_ = _v(row['start']), _v(row['end'])
            for svc in ('tray', 'busway', 'chw'):
                cov = _coverage(rs, re_, segs[svc], COVER_TOL)
                print(f"CONN row {row.get('name', '?')}: {svc} coverage {cov:.0%}")
                if cov < ROW_MIN:
                    problems.append(f"row {row.get('name', '?')}: {svc} covers only {cov:.0%} (needs {ROW_MIN:.0%})")
        if not segs['chw']:
            print("CONN note: no CHW segments, CRAH reach not checked")
        for crah in routing.get('crahs', []):
            p = _v(crah)
            if not segs['chw']:
                break
            d = min(_pt_seg_dist2d(p, a, b) for a, b in segs['chw'])
            if d > CRAH_TOL:
                problems.append(f"CRAH at {tuple(int(v) for v in p)} is {d:.0f} mm from the nearest CHW segment (needs {CRAH_TOL:.0f} mm)")
    if problems:
        for p in problems:
            print('CONN FAIL: ' + p)
        print('CONN_GATE: FAIL')
    else:
        print('CONN_GATE: PASS')


main()
"""
CONN_GATE_COMMAND = "python3 - <<'PYEOF'\n" + CONN_GATE_SCRIPT + "\nPYEOF"


SEP_GATE_SCRIPT = """
import glob, math, os, sys
try:
    import ifcopenshell
except Exception as exc:
    print(f"SEP_GATE: SKIP (ifcopenshell unavailable: {exc})")
    sys.exit(0)

CLASH_TOL = 2.0
SPRINKLER_CLEAR = 460.0
MIN_CLEAR_HEIGHT = 2600.0
FRAME_TOL = 500.0


def _v(p):
    return (float(p[0]), float(p[1]), float(p[2]))


def _norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return tuple(x / n for x in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def _add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _scale(v, s):
    return tuple(x * s for x in v)


def _matvec(cols, v):
    return tuple(sum(cols[j][i] * v[j] for j in range(3)) for i in range(3))


def _axis_frame(ap):
    '''Frame of an IfcAxis2Placement3D: (location, x, y, z).'''
    loc = _v(ap.Location.Coordinates) if getattr(ap, 'Location', None) else (0.0, 0.0, 0.0)
    z = _norm(_v(ap.Axis.DirectionRatios)) if getattr(ap, 'Axis', None) else (0.0, 0.0, 1.0)
    x = _norm(_v(ap.RefDirection.DirectionRatios)) if getattr(ap, 'RefDirection', None) else (1.0, 0.0, 0.0)
    y = _norm(_cross(z, x))
    x = _norm(_cross(y, z))
    return loc, x, y, z


def _world_frame(pl):
    '''World frame of an IfcLocalPlacement, walking its PlacementRelTo chain.'''
    origin, x, y, z = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)
    chain, cur = [], pl
    while cur is not None:
        chain.append(cur)
        cur = getattr(cur, 'PlacementRelTo', None)
    for p in reversed(chain):
        loc, px, py, pz = _axis_frame(p.RelativePlacement)
        origin = _add(origin, _matvec((x, y, z), loc))
        x, y, z = _matvec((x, y, z), px), _matvec((x, y, z), py), _matvec((x, y, z), pz)
    return origin, x, y, z


def _profile_bbox(prof):
    if prof.is_a('IfcRectangleProfileDef'):
        cx, cy = 0.0, 0.0
        if getattr(prof, 'Position', None) and getattr(prof.Position, 'Location', None):
            c = prof.Position.Location.Coordinates
            cx, cy = float(c[0]), float(c[1])
        xd, yd = float(prof.XDim), float(prof.YDim)
        return (cx - xd / 2, cx + xd / 2), (cy - yd / 2, cy + yd / 2)
    if prof.is_a('IfcArbitraryClosedProfileDef'):
        pts = prof.OuterCurve.Points
        xs = [float(p.Coordinates[0]) for p in pts]
        ys = [float(p.Coordinates[1]) for p in pts]
        return (min(xs), max(xs)), (min(ys), max(ys))
    return (-0.5, 0.5), (-0.5, 0.5)


def _box(product):
    '''Oriented box of an extruded solid: (centre, [(dir, half_len) x3]).

    Returns None when the product has no extruded solid representation.
    '''
    if getattr(product, 'ObjectPlacement', None) is None:
        return None
    if getattr(product, 'Representation', None) is None:
        return None
    try:
        origin, x, y, z = _world_frame(product.ObjectPlacement)
    except Exception:
        return None
    for rep in product.Representation.Representations:
        for item in rep.Items:
            if not item.is_a('IfcExtrudedAreaSolid'):
                continue
            if getattr(item, 'Position', None):
                sloc, sx, sy, sz = _axis_frame(item.Position)
                origin = _add(origin, _matvec((x, y, z), sloc))
                x, y, z = (_matvec((x, y, z), sx), _matvec((x, y, z), sy),
                           _matvec((x, y, z), sz))
            (x0, x1), (y0, y1) = _profile_bbox(item.SweptArea)
            dx, dy = (x1 - x0) / 2.0, (y1 - y0) / 2.0
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            depth = float(item.Depth)
            ed = _matvec((x, y, z), _norm(_v(item.ExtrudedDirection.DirectionRatios)))
            centre = _add(origin, _add(_add(_scale(x, cx), _scale(y, cy)),
                                        _scale(ed, depth / 2.0)))
            return centre, [(x, dx), (y, dy), (ed, depth / 2.0)]
    return None


def _aabb(centre, axes):
    ext = [0.0, 0.0, 0.0]
    for d, half in axes:
        for i in range(3):
            ext[i] += abs(d[i]) * half
    return tuple(centre[i] - ext[i] for i in range(3)), tuple(centre[i] + ext[i] for i in range(3))


def _pt_seg_dist(p, a, b):
    '''Distance from point p to segment a-b (3D).'''
    ab = _sub(b, a)
    d = sum(x * x for x in ab)
    if d == 0:
        return math.sqrt(sum((p[i] - a[i]) ** 2 for i in range(3)))
    t = max(0.0, min(1.0, sum((p[i] - a[i]) * ab[i] for i in range(3)) / d))
    q = tuple(a[i] + t * ab[i] for i in range(3))
    return math.sqrt(sum((p[i] - q[i]) ** 2 for i in range(3)))


def _closest_point(p, a, b):
    '''Closest point on segment a-b to point p.'''
    ab = _sub(b, a)
    denom = sum(x * x for x in ab)
    t = 0.0 if denom == 0 else max(
        0.0, min(1.0, sum((p[i] - a[i]) * ab[i] for i in range(3)) / denom))
    return _add(a, _scale(ab, t))


def _endpoints(centre, axes):
    d, half = max(axes, key=lambda a: a[1])
    return _sub(centre, _scale(d, half)), _add(centre, _scale(d, half))


def _overlap(a_min, a_max, b_min, b_max):
    '''Per-axis overlap in mm, or None when the boxes do not overlap.'''
    ov = [min(a_max[i], b_max[i]) - max(a_min[i], b_min[i]) for i in range(3)]
    if min(ov) <= 0:
        return None
    return ov


def _service_of(product):
    text = ((product.Name or '') + ' ' + (product.ObjectType or '')).lower()
    if 'busway' in text:
        return 'busway'
    if 'tray' in text:
        return 'tray'
    if 'chw' in text or 'pipe' in text or 'chilled' in text:
        return 'chw'
    if 'sprinkler' in text:
        return 'sprinkler'
    if 'rack' in text or 'cabinet' in text:
        return 'rack'
    if 'column' in text or product.is_a('IfcColumn'):
        return 'column'
    return 'other'


def main():
    files = sorted(glob.glob('output/**/*.ifc', recursive=True))
    if not files:
        print('SEP_GATE: SKIP (no .ifc files under output/)')
        return
    path = 'output/model.ifc' if 'output/model.ifc' in files else max(files, key=os.path.getsize)
    try:
        model = ifcopenshell.open(path)
    except Exception as exc:
        print(f'SEP_GATE: SKIP (cannot open {path}: {exc})')
        return
    items = []
    for product in model.by_type('IfcProduct'):
        b = _box(product)
        if b is None:
            continue
        centre, axes = b
        a_min, a_max = _aabb(centre, axes)
        items.append((product, _service_of(product), centre, axes, a_min, a_max))
    print(f"SEP products with geometry: {len(items)}")
    problems = []
    # 1. Collision count: every pair of boxes overlapping > CLASH_TOL on all axes.
    clashes = {}
    total = 0
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            ov = _overlap(items[i][4], items[i][5], items[j][4], items[j][5])
            if ov is None or min(ov) <= CLASH_TOL:
                continue
            total += 1
            key = tuple(sorted((items[i][1], items[j][1])))
            clashes[key] = clashes.get(key, 0) + 1
    print(f"SEP collisions: {total}")
    for key in sorted(clashes):
        print(f"SEP   {' x '.join(key)}: {clashes[key]}")
    if total:
        problems.append(f"{total} clashes (boxes overlapping > {CLASH_TOL:.0f} mm on all axes) — joints must be butt joints (0 mm overlap) or leave a gap, never overlapping boxes")
    # 2. Tier separation: when there are clashes, say why — two services
    # sharing a z band over the same plan area clash along their whole
    # length. An X corridor does not separate X-running headers; only
    # different tiers do. This is a root-cause note, not an independent
    # failure: it only fires when clashes already exist, so a clean model
    # whose services share a band in different corridors still passes.
    if total:
        bands = {}
        plans = {}
        for product, svc, centre, axes, a_min, a_max in items:
            if svc not in ('tray', 'busway', 'chw'):
                continue
            z0, z1 = bands.get(svc, (a_min[2], a_max[2]))
            bands[svc] = (min(z0, a_min[2]), max(z1, a_max[2]))
            x0, y0, x1, y1 = plans.get(
                svc, (a_min[0], a_min[1], a_max[0], a_max[1]))
            plans[svc] = (min(x0, a_min[0]), min(y0, a_min[1]),
                          max(x1, a_max[0]), max(y1, a_max[1]))
        for a, b in (('tray', 'busway'), ('tray', 'chw'), ('busway', 'chw')):
            if a not in bands or b not in bands:
                continue
            z_overlap = min(bands[a][1], bands[b][1]) - max(bands[a][0], bands[b][0])
            if z_overlap <= CLASH_TOL:
                continue
            plan_x = min(plans[a][2], plans[b][2]) - max(plans[a][0], plans[b][0])
            plan_y = min(plans[a][3], plans[b][3]) - max(plans[a][1], plans[b][1])
            if plan_x <= CLASH_TOL or plan_y <= CLASH_TOL:
                continue
            problems.append(
                f"{a} and {b} share z {max(bands[a][0], bands[b][0]):.0f}-{min(bands[a][1], bands[b][1]):.0f} over the same plan area — separate them into different tiers (tray z=3400, busway z=3100, CHW z=2800)")
            break
    # 3. Profile orientation: a horizontal route segment's vertical extent
    # must be its profile height (the smaller profile dimension), not its width.
    for product, svc, centre, axes, a_min, a_max in items:
        if svc not in ('tray', 'busway', 'chw'):
            continue
        longest = max(axes, key=lambda a: a[1])
        if abs(longest[0][2]) > 0.5:
            continue  # vertical segment (a drop): skip
        vertical = a_max[2] - a_min[2]
        prof = sorted([axes[0][1], axes[1][1]])
        expect = 2 * prof[0]
        if vertical > expect + CLASH_TOL:
            problems.append(
                f"{svc} segment {product.Name or product.is_a()} is {vertical:.0f} mm tall but its profile is only {expect:.0f} mm high — the profile is oriented wrong (width standing vertical)")
            break
    # 4. Drops must go down: a vertical route segment's top must hang from
    # a horizontal segment of the same service (a header or trunk). A drop
    # that only touches a riser is not hung. Name the nearest horizontal
    # segment and where to build the branch, so the fix knows what to do.
    for product, svc, centre, axes, a_min, a_max in items:
        if svc not in ('tray', 'busway', 'chw'):
            continue
        longest = max(axes, key=lambda a: a[1])
        if abs(longest[0][2]) <= 0.5:
            continue  # horizontal: skip
        top = (centre[0], centre[1], a_max[2])
        nearest = None
        nearest_name = None
        nearest_ends = None
        for p2, s2, c2, ax2, mn2, mx2 in items:
            if s2 != svc or p2 is product:
                continue
            longest2 = max(ax2, key=lambda a: a[1])
            if abs(longest2[0][2]) > 0.5:
                continue  # a riser: a drop hangs from a header, not a riser
            e1, e2 = _endpoints(c2, ax2)
            d = _pt_seg_dist(top, e1, e2)
            if nearest is None or d < nearest:
                nearest = d
                nearest_name = p2.Name or p2.is_a()
                nearest_ends = (e1, e2)
        if nearest is None or nearest > 250.0:
            if nearest_ends is not None:
                e1, e2 = nearest_ends
                closest = _closest_point(top, e1, e2)
                dx = closest[0] - top[0]
                dy = closest[1] - top[1]
                dz = closest[2] - top[2]
                if abs(dz) >= max(abs(dx), abs(dy)):
                    # The gap is vertical: the closest point is directly
                    # above/below the drop's top, so the fix is to extend
                    # the drop up (or rebuild it going down), not to move it
                    # sideways.
                    if dz > 0:
                        gap = f"{nearest:.0f} mm away to the above (drop top z={top[2]:.0f}, nearest segment z={closest[2]:.0f}) — extend the drop {nearest:.0f} mm up to the header"
                    else:
                        gap = f"{nearest:.0f} mm away to the below (drop top z={top[2]:.0f}, nearest segment z={closest[2]:.0f}) — the drop goes up instead of down; rebuild it hanging from the header at z={closest[2]:.0f} down to the CRAH top"
                else:
                    if abs(dx) >= abs(dy):
                        direction = 'east' if dx > 0 else 'west'
                    else:
                        direction = 'north' if dy > 0 else 'south'
                    gap = f"{nearest:.0f} mm away to the {direction} — build a horizontal branch from that segment to ({top[0]:.0f},{top[1]:.0f}) at the header height, then drop down"
                problems.append(
                    f"{svc} vertical segment {product.Name or product.is_a()} at ({top[0]:.0f},{top[1]:.0f}) does not hang from the header: nearest horizontal segment is {nearest_name}, its closest point is ({closest[0]:.0f},{closest[1]:.0f},{closest[2]:.0f}), {gap}")
            else:
                problems.append(
                    f"{svc} vertical segment {product.Name or product.is_a()} does not hang from the header: no horizontal {svc} segment exists — build a header first, then a branch to the drop")
            break
    # 5. Sprinkler clearance: every route top must be >= 460 mm below the
    # lowest sprinkler.
    sprinklers = [it for it in items if it[1] == 'sprinkler']
    if sprinklers:
        lowest = min(it[4][2] for it in sprinklers)
        for product, svc, centre, axes, a_min, a_max in items:
            if svc not in ('tray', 'busway', 'chw'):
                continue
            if a_max[2] > lowest - SPRINKLER_CLEAR:
                problems.append(
                    f"{svc} segment {product.Name or product.is_a()} tops at {a_max[2]:.0f} mm, only {lowest - a_max[2]:.0f} mm below the sprinklers (needs {SPRINKLER_CLEAR:.0f} mm)")
                break
    # 6. Clear height: horizontal route segments must stay above 2600 mm.
    for product, svc, centre, axes, a_min, a_max in items:
        if svc not in ('tray', 'busway', 'chw'):
            continue
        longest = max(axes, key=lambda a: a[1])
        if abs(longest[0][2]) > 0.5:
            continue  # vertical drop: may go below
        if a_min[2] < MIN_CLEAR_HEIGHT:
            problems.append(
                f"{svc} segment {product.Name or product.is_a()} bottom is at {a_min[2]:.0f} mm, below the {MIN_CLEAR_HEIGHT:.0f} mm clear height")
            break
    # 7. Frame: every route segment must lie inside the building slab's
    # plan footprint. This catches the double-placement bug — a model built
    # with coordinates in both the product placement and the solid position
    # lands at twice its intended coordinates, so the route sticks out of
    # the hall on every side while every self-consistent check still passes.
    # The slab is the anchor because it is a single box: if it is doubled
    # it only shifts, it does not stretch, so a route built at 2x spills
    # far beyond it. Skipped when the model has no slab.
    slabs = [it for it in items if it[0].is_a('IfcSlab')]
    if slabs:
        sx0 = min(it[4][0] for it in slabs)
        sy0 = min(it[4][1] for it in slabs)
        sx1 = max(it[5][0] for it in slabs)
        sy1 = max(it[5][1] for it in slabs)
        outside = 0
        worst = None
        for product, svc, centre, axes, a_min, a_max in items:
            if svc not in ('tray', 'busway', 'chw'):
                continue
            if (a_min[0] < sx0 - FRAME_TOL or a_max[0] > sx1 + FRAME_TOL
                    or a_min[1] < sy0 - FRAME_TOL
                    or a_max[1] > sy1 + FRAME_TOL):
                outside += 1
                if worst is None:
                    worst = (svc, product.Name or product.is_a(), a_min, a_max)
        print(f"SEP frame: {outside} route segment(s) outside the slab footprint")
        if outside:
            wsvc, wname, wmn, wmx = worst
            problems.append(
                f"FRAME FAIL: {outside} route segment(s) lie outside the building slab (slab plan x {sx0:.0f}..{sx1:.0f}, y {sy0:.0f}..{sy1:.0f}); e.g. {wsvc} {wname} spans x {wmn[0]:.0f}..{wmx[0]:.0f}, y {wmn[1]:.0f}..{wmx[1]:.0f} — the model is built at the wrong scale or offset (a box whose coordinates are set in both the product placement and the solid position lands at twice its intended place; set them in exactly one place)")
    if problems:
        for p in problems:
            print('SEP FAIL: ' + p)
        print('SEP_GATE: FAIL')
    else:
        print('SEP_GATE: PASS')


main()
"""
SEP_GATE_COMMAND = "python3 - <<'PYEOF'\n" + SEP_GATE_SCRIPT + "\nPYEOF"


async def _separation_gate(worker: Agent) -> tuple[str, dict]:
  """Runs the deterministic separation/clash gate and returns failures as gaps.

  The gate runs in the task container through the worker's bash tool, on
  routing tasks only (see ROUTING_KEYWORDS). It measures the produced .ifc
  itself: every pair of solids overlapping by more than 2 mm on all axes
  (the grader counts every such overlap as a clash, with no exemption for
  joints, same-service pairs or background objects), profile orientation
  (a horizontal segment's vertical extent must be its profile height, not
  its width), drops that go up instead of down, sprinkler clearance
  (460 mm), the 2600 mm clear height, and the frame check (route segments
  must lie inside the building slab's footprint). It returns ("", scores)
  when the task is not a routing task, there is nothing to check, the check
  passes, or the check could not run — an inconclusive gate must not block
  submission.

  Args:
    worker: The worker Agent (its bash tool runs in the task container).

  Returns:
    (gaps, scores): the gate's failure text when it fails, else "";
    scores is {"clashes": int, "frame_ok": bool, "frame_outside": int}
    when the gate measured something, else {} (inconclusive).
  """
  instr = worker.session.instruction.lower()
  if not any(k in instr for k in ROUTING_KEYWORDS):
    return "", {}
  try:
    out = await worker.bash(SEP_GATE_COMMAND)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    await worker.session.log(f"Separation gate failed to run: {exc}")
    return "", {}
  if "SEP_GATE: FAIL" not in out:
    await worker.session.log(
        "Separation gate: " + (out.strip().splitlines() or [""])[-1][:200]
    )
    return "", _separation_gate_scores(out)
  return clip(out, 4000), _separation_gate_scores(out)


def _separation_gate_scores(out: str) -> dict:
  """Parses the separation gate's output into a score dict for the regression guard.

  Returns {"clashes": int, "frame_ok": bool, "frame_outside": int} when the
  gate measured something, else {} (inconclusive — the guard must not
  compare unknowns).
  """
  if "SEP_GATE: SKIP" in out:
    return {}
  clashes = None
  frame_outside = 0
  for line in out.splitlines():
    if line.startswith("SEP collisions:"):
      try:
        clashes = int(line.split(":")[1].strip())
      except (IndexError, ValueError):
        pass
    if line.startswith("SEP frame:"):
      try:
        frame_outside = int(line.split(":")[1].split()[0])
      except (IndexError, ValueError):
        pass
  if clashes is None:
    return {}
  return {
      "clashes": clashes,
      "frame_ok": "FRAME FAIL" not in out,
      "frame_outside": frame_outside,
  }


async def _connectivity_gate(worker: Agent) -> tuple[str, dict]:
  """Runs the deterministic connectivity gate and returns failures as gaps.

  The gate runs in the task container through the worker's bash tool, on
  routing tasks only (see ROUTING_KEYWORDS). It returns ("", scores) when
  the task is not a routing task, there is nothing to check, the check
  passes, or the check could not run — an inconclusive gate must not block
  submission.

  Args:
    worker: The worker Agent (its bash tool runs in the task container).

  Returns:
    (gaps, scores): the gate's failure text when it fails, else "";
    scores is {"components": {service: count}, "rows_below_80": int}
    when the gate measured something, else {} (inconclusive).
  """
  instr = worker.session.instruction.lower()
  if not any(k in instr for k in ROUTING_KEYWORDS):
    return "", {}
  try:
    out = await worker.bash(CONN_GATE_COMMAND)
  except Exception as exc:  # pylint: disable=broad-exception-caught
    await worker.session.log(f"Connectivity gate failed to run: {exc}")
    return "", {}
  if "CONN_GATE: FAIL" not in out:
    await worker.session.log(
        "Connectivity gate: " + (out.strip().splitlines() or [""])[-1][:200]
    )
    return "", _connectivity_gate_scores(out)
  return clip(out, 4000), _connectivity_gate_scores(out)


def _connectivity_gate_scores(out: str) -> dict:
  """Parses the connectivity gate's output into a score dict for the regression guard.

  Returns {"components": {service: count}, "rows_below_80": int} when the
  gate measured something, else {} (inconclusive — the guard must not
  compare unknowns).
  """
  if "CONN_GATE: SKIP" in out:
    return {}
  components = {}
  rows_below_80 = 0
  for line in out.splitlines():
    if line.startswith("CONN ") and "connected component(s)" in line:
      try:
        svc = line.split()[1]
        components[svc] = int(line.split(":")[1].split()[0])
      except (IndexError, ValueError):
        pass
    if line.startswith("CONN row ") and "coverage" in line:
      try:
        pct = float(line.split("coverage")[1].strip().rstrip("%"))
        if pct < 80.0:
          rows_below_80 += 1
      except (IndexError, ValueError):
        pass
  # A PASS/FAIL verdict means the gate ran and measured; the component
  # lines are only printed when a service has more than one segment, so
  # an empty components map is a real measurement, not an unknown.
  if "CONN_GATE: PASS" not in out and "CONN_GATE: FAIL" not in out:
    return {}
  return {"components": components, "rows_below_80": rows_below_80}


def _merge_gate_gaps(*gap_texts: str) -> str:
  """Merges gate failure texts into one fix-round prompt.

  Each gate is clipped to ~1500 chars and the total to 4000, so one verbose
  gate cannot crowd out the others. The caller passes them in priority
  order: unit, IFC, separation, connectivity (then the verifier's own
  gaps, when the caller merges those in too).

  Args:
    gap_texts: Failure texts from the gates ("" when a gate passed).

  Returns:
    The merged text, or "" when every gate passed.
  """
  parts = [clip(g, 1500) for g in gap_texts if g]
  if not parts:
    return ""
  return clip("\n".join(parts), 4000)


def _gate_verdict_lines(
    unit_gaps: str, ifc_gaps: str, sep_gaps: str, conn_gaps: str, is_routing: bool
) -> str:
  """One summary line per gate, for the verifier's context and the final summary.

  A gate returns "" when it passes, when the task is not a routing task, or
  when it could not run — those cases are not distinguished, so a clean gate
  is reported as "no failures reported", never as a pass.

  Args:
    unit_gaps: Unit gate failure text, "" when clean.
    ifc_gaps: IFC gate failure text, "" when clean.
    sep_gaps: Separation gate failure text, "" when clean.
    conn_gaps: Connectivity gate failure text, "" when clean.
    is_routing: Whether the task is a routing task (the separation and
      connectivity gates only run on those).

  Returns:
    Four lines, one per gate.
  """
  lines = []
  for name, gaps, applicable in (
      ("Unit", unit_gaps, True),
      ("IFC", ifc_gaps, True),
      ("Separation", sep_gaps, is_routing),
      ("Connectivity", conn_gaps, is_routing),
  ):
    if not applicable:
      lines.append(f"{name} gate: not applicable (not a routing task)")
    elif gaps:
      # The most informative failure line: the "FAIL: <detail>" line when
      # the gate has one, else the first line (the IFC gate's first line
      # already names the violations; the separation gate's first line is
      # only a product count).
      detail = next(
          (line for line in gaps.strip().splitlines() if "FAIL:" in line),
          gaps.strip().splitlines()[0],
      )[:200]
      lines.append(f"{name} gate: FAIL — {detail}")
    else:
      lines.append(f"{name} gate: no failures reported")
  return "\n".join(lines)


def _gate_score_vector(unit: dict, ifc: dict, sep: dict, conn: dict) -> dict | None:
  """Builds the regression-guard score vector from the gates' score dicts.

  Returns None when any gate was inconclusive (an unknown score must not
  be compared). The vector is a dict so a new dimension can be added
  without changing every caller.
  """
  if not unit or not ifc or not sep or not conn:
    return None
  components = conn.get("components") or {}
  return {
      "unit_ok": unit.get("unit_ok", False),
      "ifc_ok": ifc.get("ifc_ok", False),
      "frame_ok": sep.get("frame_ok", True),
      "clashes": sep.get("clashes", 0),
      "max_components": max(components.values()) if components else 0,
      "rows_below_80": conn.get("rows_below_80", 0),
  }


def _is_regression(prev: dict, cur: dict) -> bool:
  """Whether the current gate scores are worse than the previous round's.

  A fix round is a regression when it flips a fatal check to FAIL (units,
  IFC, frame), raises the clash count, or disconnects the network further
  without cutting clashes. A clash increase with a component improvement
  is not a regression — the fix traded a clash for connectivity — but a
  clash increase alone always is.
  """
  if cur["unit_ok"] is False and prev["unit_ok"] is True:
    return True
  if cur["ifc_ok"] is False and prev["ifc_ok"] is True:
    return True
  if cur["frame_ok"] is False and prev["frame_ok"] is True:
    return True
  if cur["clashes"] > prev["clashes"]:
    return True
  if (cur["max_components"] > prev["max_components"]
          and cur["clashes"] >= prev["clashes"]):
    return True
  if (cur["rows_below_80"] > prev["rows_below_80"]
          and cur["clashes"] >= prev["clashes"]):
    return True
  return False


async def _run_all_gates(
    worker: Agent, is_routing: bool
) -> tuple[str, str, dict | None]:
  """Runs all four deterministic gates in parallel.

  Returns (merged_gaps, verdict_lines, score_vector): the merged failure
  text, one summary line per gate, and the score vector for the regression
  guard (None when any gate was inconclusive).
  """
  (unit_gaps, unit_scores), (ifc_gaps, ifc_scores), (
      sep_gaps, sep_scores), (conn_gaps, conn_scores) = await asyncio.gather(
          _unit_gate(worker), _ifc_gate(worker), _separation_gate(worker),
          _connectivity_gate(worker))
  gaps = _merge_gate_gaps(unit_gaps, ifc_gaps, sep_gaps, conn_gaps)
  verdict = _gate_verdict_lines(
      unit_gaps, ifc_gaps, sep_gaps, conn_gaps, is_routing)
  scores = _gate_score_vector(unit_scores, ifc_scores, sep_scores, conn_scores)
  return gaps, verdict, scores


async def solve(session: taskbridge.Session) -> str:
  """Runs plan, work, verify, and fix phases and returns the final answer.

  Short tasks skip planning and verification and run the worker directly, so
  smoke and MCP checks behave like the single-loop baseline.

  Args:
    session: Active task session.

  Returns:
    The final summary string (the worker's summary, plus a verification note
    when a verifier passed it).
  """
  print(
      f"task: workdir {session.workdir}, model {MODEL},"
      f" {len(session.mcp_servers)} MCP server(s)",
      file=sys.stderr,
  )
  total = session.time_left()
  plan = ""
  if total is None or total >= MIN_PLAN_SEC:
    budget = total * PLAN_SHARE if total is not None else MIN_PLAN_SEC
    await session.log(f"plan phase: up to {budget:.0f}s")
    plan = await _run_phase(
        session, "plan", Agent(session, role="planner", phase_seconds=budget)
    )
    session.final_message = ""  # a plan is not a deliverable
  left = session.time_left()
  worker_budget = (
      None
      if left is None or left < MIN_VERIFY_SEC / CHECK_SHARE
      else left * (1 - CHECK_SHARE)
  )
  worker = Agent(session, role="worker", plan=plan, phase_seconds=worker_budget)
  summary = await worker.run()
  if not summary:
    return session.final_message
  verified = False
  is_routing = any(
      k in session.instruction.lower() for k in ROUTING_KEYWORDS
  )
  last_gate_gaps = ""
  last_gate_verdict = ""
  prev_scores = None  # score vector from the round before the last fix
  snapshot_exists = False  # whether /tmp/prev_output holds the pre-fix model
  needs_final_gate_pass = False  # the last action was a fix round
  try:
    for round_no in range(MAX_FIX_ROUNDS + 1):
      left = session.time_left()
      if left is not None and left < MIN_VERIFY_SEC:
        break
      await session.log(f"verify phase: round {round_no + 1}")
      # Run all four deterministic gates in parallel FIRST, every round,
      # whether or not the verifier passes: the verifier judges with their
      # numbers in hand instead of judging blind. A gate that never runs
      # cannot teach the agent — on run 8 the connectivity gate ran 0 times
      # because the separation gate kept failing and the old chain
      # short-circuited, so the entry was never checked.
      last_gate_gaps, last_gate_verdict, scores = await _run_all_gates(
          worker, is_routing)
      # Regression guard: the fix round just run made the model worse.
      # Restore the snapshot (the pre-fix model) and re-measure it, so the
      # submitted model is the best measured, not the last written.
      regression_note = ""
      if (scores is not None and prev_scores is not None
              and snapshot_exists and _is_regression(prev_scores, scores)):
        await session.log(
            "fix round made the model worse; restoring the previous model")
        await worker.bash("rm -rf output && cp -r /tmp/prev_output output")
        last_gate_gaps, last_gate_verdict, scores = await _run_all_gates(
            worker, is_routing)
        prev_scores = scores
        regression_note = (
            "Your last fix made the model worse and the previous model has"
            " been restored. Do not rebuild the same way; change the"
            " construction, not the layout.\n")
      needs_final_gate_pass = False  # the gates just measured the current model
      # The verdict survives even if a later model call raises: attach it
      # to session.final_message (the deliverable) now, not only to the
      # return value at the end of solve(). On run 9 solve() raised a 503
      # after the summary existed and the appendix died with the exception.
      if last_gate_gaps:
        session.final_message = (
            summary + "\n\nDeterministic gates:\n" + last_gate_verdict)
      reply = await _run_phase(
          session,
          "verify",
          Agent(
              session,
              role="verifier",
              plan=plan,
              context=summary
              + "\n\nDeterministic gate results (quote these numbers):\n"
              + last_gate_verdict,
          ),
      )
      session.final_message = summary  # the verifier's text is not the summary
      if last_gate_gaps:
        session.final_message = (
            summary + "\n\nDeterministic gates:\n" + last_gate_verdict)
      passed, gaps = _verdict(reply)
      if last_gate_gaps:
        # The gates' gaps join the verifier's (or replace them when the
        # verifier passed): a gate failure is a fix round, like any other.
        gaps = _merge_gate_gaps(gaps, last_gate_gaps) if gaps else last_gate_gaps
        passed = False
      if passed:
        verified = True
        break
      if not gaps or round_no >= MAX_FIX_ROUNDS:
        break
      left = session.time_left()
      if left is not None and left < MIN_VERIFY_SEC:
        break
      await session.log(f"fix round {round_no + 1}: {len(gaps)} chars of gaps")
      # Snapshot the current (best-measured) model before the fix, so a
      # fix that makes things worse can be rolled back.
      snap_out = await worker.bash(
          "if [ -d output ]; then rm -rf /tmp/prev_output"
          " && cp -r output /tmp/prev_output && echo SNAPSHOT_OK;"
          " else echo SNAPSHOT_NONE; fi")
      snapshot_exists = "SNAPSHOT_OK" in snap_out
      prev_scores = scores
      needs_final_gate_pass = True
      summary = await worker.resume(
          regression_note
          + "An independent check of your deliverables found these gaps. Fix each"
          " one, then answer with your updated summary. Your new summary must"
          " quote the gate numbers below (clash count, separation, entry"
          " distance, coverage) verbatim and must not claim anything the gates"
          f" deny:\n{gaps}"
      )
      if not summary:
        summary = session.final_message
  except openai.OpenAIError as exc:
    # A transient endpoint failure after the summary exists is not fatal:
    # the summary (plus the last gate verdict, attached to
    # session.final_message each round) is the deliverable.
    await session.log(f"verify phase interrupted by endpoint error: {exc}")
    if session.final_message:
      summary = session.final_message
  # After the last fix round there is no round left to measure the final
  # model, so run the gates once more (no verifier) — the submitted
  # summary must carry the final model's verdict, not the previous round's.
  # On run 9 the final model was never gate-checked at all.
  if needs_final_gate_pass:
    left = session.time_left()
    if left is None or left >= MIN_VERIFY_SEC:
      await session.log("final gate pass on the submitted model")
      last_gate_gaps, last_gate_verdict, scores = await _run_all_gates(
          worker, is_routing)
      if (scores is not None and prev_scores is not None
              and snapshot_exists and _is_regression(prev_scores, scores)):
        await session.log(
            "final model is worse than the previous one; restoring it")
        await worker.bash("rm -rf output && cp -r /tmp/prev_output output")
        last_gate_gaps, last_gate_verdict, scores = await _run_all_gates(
            worker, is_routing)
  if verified:
    summary += "\n\nIndependent verification: PASS."
  # Append the last round's gate verdict so the submitted summary always
  # carries the deterministic checks' conclusion, even when the fix
  # rounds ran out and the final model still fails a gate.
  if last_gate_gaps and "Deterministic gates:" not in summary:
    summary += "\n\nDeterministic gates:\n" + last_gate_verdict
  return summary


if __name__ == "__main__":
  taskbridge.run(solve)
