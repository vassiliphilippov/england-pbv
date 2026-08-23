# Python Style Guide

This document is normative for all Python code in this repository. It covers general Python
conventions, typing, and data handling.

It is adapted from the `glite-english-audit` / `glite-arf` Python style guides (Apache-2.0).
Sections about transcript privacy, the diagnostics registry, and `artifacts/io` helpers are
inherited conventions: apply their spirit (never leak raw dataset payloads into logs; write
artifacts atomically; report problems with stable identifiers) even though this project has no
user transcripts. Where this guide and tool configuration disagree, the configuration in
`pyproject.toml` wins; report the conflict instead of coding around it.

## Quick Reference

### Frequently Ignored Rules

These rules are well-established but often overlooked:

1. **Use dataclasses instead of tuples** - Never use `tuple[int, Path, str]`, always use
   dataclasses or Pydantic models
2. **Use keyword arguments** - Functions with 2+ heterogeneous params require keyword arguments
3. **Never return tuples** - Functions must return dataclasses, not tuples
4. **Centralize paths** - All runtime file locations come from `england_pbv.paths`;
   never hardcode a path
5. **Use constants for magic strings** - No hardcoded strings like `"claude_code"`, `"completed"`
6. **Use `None` for missing data** - Never use `0.0` or `""` when data is not available;
   `None`/`null` means "no measurement", `0.0` means "measured zero"
7. **Never emit source text** - Discovery and inventory code must not print, log, or return the
   text of user transcripts; summaries and counts only

### Core Principles

* **Type Safety**: Use enums, dataclasses, and explicit types throughout
* **Constants Over Magic Values**: Extract all reused strings/values to typed constants
* **Separation of Concerns**: Keep validation, conversion, and business logic separate
* **Explicit Over Implicit**: Be explicit about types and conversions
* **Resource Safety**: Always clean up temporary files with try/finally
* **Privacy by Construction**: Code paths that touch user text must make leaking it impossible,
  not merely unlikely

* * *

## Toolchain

The project uses one toolchain. Do not substitute alternatives.

* **Python 3.12+** is required. Use 3.12 syntax (see [Python Version](#python-version)).
* **uv** manages the environment and runs every command. Never call `pip`, `python`, or a bare
  `pytest` directly; always go through `uv run`.
* **ruff** is both formatter and linter (line length 100; rules `E,F,W,I,UP,B,SIM,RUF`).
* **mypy --strict** must be clean for `src` and `tests`.
* **pytest** is the test runner.

Run the full gate before finishing any change:

```bash
uv sync --locked --all-groups
uv run ruff format .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run python -m england_pbv.verification.verify_skills
uv run python -m england_pbv.artifacts.schema_export --check
```

A change is not done while any of these fail. Do not silence a checker with `# type: ignore`,
`# noqa`, or a config exclusion to make the gate pass; fix the code. If a suppression is truly
unavoidable, scope it to one line and state the reason in a comment.

* * *

## General Coding Practices

### Use `@property` only for simple operations

Never put non-trivial computation or IO inside a `@property`.

#### Why:

It's impossible to distinguish an innocent field access from a heavy/IO `@property` access on the
call site, which becomes a problem in loops or tight timing scenarios.

#### Do:

```python
class RunView:
    utterance_count: int

    @property
    def has_utterances(self) -> bool:
        return self.utterance_count > 0

    def load_manifest(self) -> RunManifest:
        return read_manifest_from_disk(run_id=self.run_id)
```

#### Don't:

```python
class RunView:
    utterance_count: int

    @property
    def has_utterances(self) -> bool:
        return self.utterance_count > 0

    @property
    def manifest(self) -> RunManifest:
        return read_manifest_from_disk(run_id=self.run_id)  # Hidden IO
```

* * *

### Use explicit checks instead of relying on falsiness

Don't use the idiomatic falsiness of empty lists, zeroes, empty strings, and `None`s. Check them
explicitly.

#### Why:

This idiom is too error-prone, especially in presence of `T | None` types.

#### Do:

```python
if snapshot_dir is None:
    ...

if len(utterances) == 0:
    ...

if (manifest := load_manifest(run_dir=run_dir)) is not None:
    ...
```

#### Don't:

```python
# Assuming non-bool x
if snapshot_dir:
    ...

if manifest := load_manifest(run_dir=run_dir):
    ...
```

* * *

### Use `None` for missing data, never zero or empty string

When a metric, measurement, or result is **not available** (data doesn't exist, computation was
skipped, input missing), use `None` — never `0`, `0.0`, or `""`. Zero is a valid measurement;
`None` means "no measurement was taken."

#### Why:

Using `0.0` for "not available" is indistinguishable from an actual value of zero. Downstream
consumers (reports, estimators, aggregators) will treat it as a real value — averaging it in,
comparing it, summing it. This silently corrupts results.

#### Do:

```python
@dataclass(frozen=True, slots=True)
class StageTiming:
    stage_id: StageId
    duration_seconds: float | None  # None when the stage never ran
    was_skipped: bool


# JSON output: {"duration_seconds": null} — clearly "not available"
```

#### Don't:

```python
@dataclass(frozen=True, slots=True)
class StageTiming:
    stage_id: StageId
    duration_seconds: float  # 0.0 when skipped — looks like an instant stage
    was_skipped: bool


# JSON output: {"duration_seconds": 0.0} — real timing or missing data?
```

* * *

### Put general, context-y parameters first when defining functions

A useful rule of thumb is "would it be convenient to use `partial()` on this function".

#### Why:

Consistency, extra semantic information, and convenience of `partial()`.

#### Do:

```python
def collect_instances(
    registry: AdapterRegistry,
    run_dir: Path,
    adapter_ids: list[str],
) -> None: ...
```

#### Don't:

```python
def collect_instances(
    adapter_ids: list[str],
    registry: AdapterRegistry,
    run_dir: Path,
) -> None: ...
```

* * *

### Write durations in fractional seconds (floats)

Store all durations as fractional seconds using floats.

#### Why:

Consistency. We don't normally need to be more precise than milliseconds, and milliseconds can be
perfectly expressed as fractional seconds.

#### Do:

```python
SNAPSHOT_TIMEOUT: float = 30.000  # seconds as a float
```

#### Don't:

```python
SNAPSHOT_TIMEOUT: int = 30000  # milliseconds not as a float
```

* * *

### Use "kind" instead of "type" in names

#### Why:

`type` clashes with built-in `type` too much.

#### Do:

```python
class SourceKind(Enum): ...
```

#### Don't:

```python
class SourceType(Enum): ...
```

* * *

## Type System

### Use dataclasses instead of tuples

**CRITICAL**: Never use complex tuple types for structured data or return values. Always use
dataclasses (or, at IO boundaries, Pydantic models).

#### Why:

* Type safety and IDE support
* Self-documenting code
* Prevents position-based errors
* Required by project style guide

#### Do:

```python
@dataclass(frozen=True, slots=True)
class SnapshotOutcome:
    file_count: int
    snapshot_dir: Path
    adapter_id: str


def snapshot_instance() -> SnapshotOutcome:
    return SnapshotOutcome(
        file_count=12,
        snapshot_dir=Path("snapshots") / "claude_code",
        adapter_id="claude_code",
    )
```

#### Don't:

```python
def snapshot_instance() -> tuple[int, Path, str]:
    return (12, Path("snapshots") / "claude_code", "claude_code")
```

#### Exception: Named tuples for cache keys

Plain tuples are acceptable for cache keys, but use `NamedTuple` for type safety:

```python
from typing import NamedTuple


class DedupKey(NamedTuple):
    adapter_id: str
    session_hash: str
    tokenizer_version: str


seen: dict[DedupKey, UtteranceRef] = {}
seen[DedupKey("claude_code", "ab12", "1.0.0")] = ref
```

**Don't use plain tuples even for cache keys:**

```python
seen: dict[tuple[str, str, str], UtteranceRef] = {}  # Hard to understand
seen[("claude_code", "ab12", "1.0.0")] = ref  # What does each position mean?
```

* * *

### Use enum objects internally, convert to strings only at boundaries

Store enum objects throughout internal logic. Only convert to strings at IO boundaries (JSON
serialization, CLI output). All shared enums live in `england_pbv.artifacts.enums`;
add new shared enums there instead of defining parallel copies.

Compare enum members, never their raw string values.

#### Why:

* **Type safety**: Catches typos and invalid values at type-checking time
* **IDE support**: Better autocomplete and refactoring
* **Clear intent**: The type system documents valid values
* **Separation of concerns**: Internal logic stays clean; Pydantic handles conversion once at
  the serialization boundary

#### Do:

```python
from england_pbv.artifacts.enums import Modality, TextStatus


@dataclass(frozen=True, slots=True)
class UtteranceSummary:
    modality: Modality
    status: TextStatus


if summary.modality is Modality.SPOKEN_ASR:
    ...
```

#### Don't:

```python
@dataclass(frozen=True, slots=True)
class UtteranceSummary:
    modality: str  # Lost type safety
    status: str


if summary.modality == "spoken_asr":  # Typo-prone string comparison
    ...
```

* * *

### Use semantic type aliases for domain-specific strings

Use type aliases like `AdapterId`, `RunId`, `UtteranceId` for domain-specific strings. Keep `str`
for truly generic strings.

#### Why:

* Self-documenting code
* Helps catch logical errors where different string types are mixed
* Makes function signatures clearer

#### Do:

```python
# Define semantic type aliases (Python 3.12+ syntax)
type AdapterId = str
type RunId = str
type UtteranceId = str


def collect_utterance_ids(
    records: list[NormalizedUtterance],
) -> set[UtteranceId]:  # Clear: returns a set of utterance IDs
    ...
```

#### Don't:

```python
def collect_utterance_ids(
    records: list[NormalizedUtterance],
) -> set[str]:  # Unclear: str of what?
    ...
```

**Note**: Don't overuse. Keep `str` for truly generic strings (messages, display strings,
diagnostic text).

* * *

### Use `int | None` instead of `Optional[int]`

#### Why:

Consistency with modern Python typing conventions.

#### Do:

```python
def parse_year(value: int | None) -> str | None: ...
```

#### Don't:

```python
from typing import Optional


def parse_year(value: Optional[int]) -> Optional[str]: ...
```

* * *

### Use `int | float` in `isinstance`, not `(int, float)`

#### Why:

Consistency with PEP 604 union syntax. Enforced by ruff rule UP038.

#### Do:

```python
if isinstance(value, int | float):
    ...
```

#### Don't:

```python
if isinstance(value, (int, float)):
    ...
```

* * *

### Use exhaustive matches with `assert_never`

Use `assert_never` to make mypy scream when you forget to handle a branch or element of a type
union. This applies especially to `StageStatus`, `RunStatus`, and other enums from
`artifacts/enums.py` that grow over time.

#### Why:

Types and code change over time. Exhaustive checking ensures that when unions are extended, mypy
will complain that not every case is covered.

#### Do:

```python
from typing import assert_never

from england_pbv.artifacts.enums import StageStatus


def describe(status: StageStatus) -> str:
    match status:
        case StageStatus.PENDING:
            return "not started"
        case StageStatus.IN_PROGRESS:
            return "in progress"
        case StageStatus.PRODUCED | StageStatus.VERIFIED_DETERMINISTIC:
            return "awaiting verification"
        case StageStatus.VERIFIED_SEMANTIC | StageStatus.PROMOTED:
            return "done"
        case StageStatus.QUARANTINED | StageStatus.FAILED | StageStatus.INVALIDATED:
            return "needs another pass"
        case _:
            assert_never(status)
```

#### Don't:

```python
def describe(status: StageStatus) -> str:
    if status is StageStatus.PENDING:
        return "not started"
    else:
        # If StageStatus gains a member, this silently misreports it
        return "done"
```

**Note**: You can also use `assert_never` for unreachable code, e.g. when you have early returns
that should always return a value first.

* * *

### Use `T_Whatever` format for meaningful type variables

When you need type variables that aren't just `T`, use the `T_Whatever` format.

#### Why:

Consistency across the codebase.

#### Do:

```python
T_Record = TypeVar("T_Record")
```

#### Don't:

* `RecordT`
* `Record`
* `RecordType`

Prefer PEP 695 syntax (`def load[T_Record](...) -> T_Record`) where it works with the tooling.

* * *

### Satisfy mypy and use explicit types as assertions

Prefer type inference but use explicit type annotations in three cases:

* Wherever types are required by mypy (function signatures, tricky inference)
* As assertions that the inferred type matches your intuition
* As documentation

#### Why:

Relying on type inference is concise, but sometimes the inferred type might not match intuition
and even mask an error.

#### Do:

```python
def total_tokens() -> int:
    a: int = count_prompt_tokens()  # Assert that this returns int
    b = count_reply_tokens()  # Let it infer when safe
    return a + b
```

#### Don't:

```python
def total_tokens() -> int:
    a: int = 1  # Unnecessary, obviously int
    b: int = 2
    return a + b
```

* * *

### Use the type system to encode business logic constraints

Try to model business constraints in the type system, as long as it's practical.

#### Why:

The earlier we find mistakes, the less costly they are. Using the type system lets us find errors
even before writing tests. In this project it also enforces privacy: a function that accepts a
`SafeMistakeRecord` cannot accidentally receive a `PrivateMistake` that still carries source text.

#### Do:

```python
@dataclass(frozen=True, slots=True)
class UnverifiedFinding:
    utterance_id: str
    claim: str


@dataclass(frozen=True, slots=True)
class VerifiedFinding:
    utterance_id: str
    claim: str
    verifier_version: str


def package(finding: VerifiedFinding) -> None: ...  # Cannot be called with an unverified finding
```

#### Don't:

```python
@dataclass(frozen=True, slots=True)
class Finding:
    utterance_id: str
    claim: str
    verifier_version: str | None  # None means "not verified yet" — easy to misuse


def package(finding: Finding) -> None: ...
```

* * *

### Use explicit types for variable declarations

Always provide explicit type hints for lists, dicts, and other collections, even when the values
make the type obvious.

#### Why:

* Serves as inline documentation
* Catches inference errors early
* Consistent with project style

#### Do:

```python
TEXT_FIELDS: list[str] = [
    "claim",
    "explanation",
    "category",
]
ADAPTER_IDS: list[str] = [
    "claude_code",
    "codex",
    "aider",
]
```

#### Don't:

```python
TEXT_FIELDS = ["claim", "explanation", "category"]
```

* * *

## Asserts

### Use `assert` for defensive coding and documentation

Whenever you assume something to be true, assert it explicitly with an `assert`.

#### Why:

`assert` serves two purposes:

* Surfacing broken assumptions early and explicitly
* Documenting the assumptions you're making

#### Do:

```python
def trim_nonempty_list(lst: list[T], n: int) -> list[T]:
    assert len(lst) > 0, "list is non-empty"
    assert 0 <= n <= len(lst), "n is within bounds"
    return lst[:n]
```

#### Don't:

```python
def trim_nonempty_list(lst: list[T], n: int) -> list[T]:
    return lst[:n]
```

**Note**: An assertion message must never interpolate source text or any other private value.
Assert on lengths, counts, IDs, and hashes — never on transcript content.

* * *

### Assume that `assert`s are always run

We never run Python with the `-O` flag, so `assert` can be assumed to always run. Plan expensive
checks accordingly.

#### Why:

Many Python libraries use asserts to assert things that must always be true, including
safety-critical conditions. There is little performance benefit to disabling assertions. Being
able to assert things should be encouraged.

* * *

### Use positive assertion messages

Positive assertion messages make the intent clearer.

#### Why:

Clear communication about what is expected, not what went wrong.

#### Do:

```python
assert manifest.is_complete(), "manifest is complete"
assert isinstance(record, SafeMistakeRecord), "Expected SafeMistakeRecord"
```

#### Don't:

* `assert manifest.is_complete(), "Manifest error"`
* `assert manifest.is_complete(), "manifest is not complete"`

* * *

## Constants and Magic Strings

### Replace magic strings with named constants

**CRITICAL**: Never use hardcoded string literals for field names, adapter IDs, statuses, or
comparison values used in multiple places. Values that already exist as enums in
`artifacts/enums.py` must be used as enum members, not re-declared as string constants.

#### Why:

1. **Maintainability**: Update in only one place
2. **Type Safety**: Explicit type hints catch errors early
3. **Readability**: Self-documenting
4. **Refactoring**: IDE tools work better
5. **Consistency**: Prevents typos

#### Do:

```python
# In the module that owns the concept
MANIFEST_FILENAME: str = "run-manifest.json"
UTTERANCES_FILENAME: str = "utterances.jsonl"
MAX_EVIDENCE_SPAN_CHARS: int = 200

# In usage
manifest_path = run_dir / MANIFEST_FILENAME

if stage.status is StageStatus.COMPLETED:
    ...
```

#### Don't:

```python
manifest_path = run_dir / "run-manifest.json"

if stage.status.value == "completed":
    ...
```

* * *

### Centralize runtime paths in `england_pbv.paths`

**CRITICAL**: All runtime file locations (runtime roots per OS, run directories, snapshot
directories) come from `england_pbv.paths`. Never construct a home-directory or
OS-specific path anywhere else, and never hardcode an absolute path. Filenames that only one
module uses may live as constants in that module; directory layout belongs to `paths.py`.

#### Why:

1. **Single Source of Truth**: OS differences (macOS, Linux, Windows, WSL) are handled once
2. **Easy Updates**: Change layout without touching business logic
3. **Clear Separation**: Path policy vs. business logic are separate concerns
4. **Testability**: Tests redirect the root to `tmp_path` in one place

#### Do:

```python
from england_pbv import paths

run_dir = paths.run_dir(run_id)
snapshot_dir = paths.snapshot_dir(run_id)
```

#### Don't:

```python
# OS-specific layout scattered through business logic
run_dir = Path.home() / ".england-pbv" / "runs" / run_id
```

* * *

### Report problems through registered diagnostics, not ad-hoc strings

User-visible and machine-readable problems are `Diagnostic` values created with
`Diagnostic.from_code(...)`. Every code is registered with a description in
`england_pbv.diagnostics.codes`. The registry is append-only: never reuse or renumber an
existing code; add a new one in the same change that first emits it.

#### Why:

Stable codes let tests, verifiers, and support workflows match on the code instead of parsing
prose. Free-form error strings drift and break consumers silently.

#### Do:

```python
from england_pbv.diagnostics.codes import Diagnostic

diagnostics.append(
    Diagnostic.from_code(
        "SOURCE_UNSUPPORTED_SCHEMA",
        "schema version 9 is newer than this client supports",
        item_ref=instance_key,
    )
)
```

#### Don't:

```python
errors.append("adapter failed: unsupported schema!!")  # No code, unmatchable
```

* * *

## Function Calls and Parameters

### Use keyword arguments for functions with 2+ heterogeneous parameters

**CRITICAL**: When calling functions with two or more heterogeneous parameters, always use
keyword argument syntax.

#### Why:

* More resilient to refactorings and typos
* Easier to read
* Explicitly required by project style guide

#### Do:

```python
normalize_transcript(
    transcript_path=transcript_path,
    tokenizer_version=tokenizer_version,
    min_word_count=min_word_count,
)
```

#### Don't:

```python
normalize_transcript(transcript_path, tokenizer_version, min_word_count)
```

**Exception**: Homogeneous parameters (all same type) don't require keyword arguments:

```python
merge_inventories(inv1, inv2, inv3)  # All same type, OK
```

* * *

### Use multi-line format for functions with 2+ parameters

When defining functions with more than two parameters, write each parameter on its own line with a
trailing comma.

#### Why:

* Better readability
* Easier diffs in version control
* Consistent formatting

#### Do:

```python
def normalize_transcript(
    transcript_path: Path,
    tokenizer_version: str,
    min_word_count: int,
) -> list[NormalizedUtterance]: ...
```

#### Don't:

```python
def normalize_transcript(
    transcript_path: Path, tokenizer_version: str, min_word_count: int
) -> list[NormalizedUtterance]: ...
```

* * *

## Pydantic and Data Validation

### Use Pydantic BaseModel for JSON files with schemas you control

Use Pydantic `BaseModel` for reading and writing JSON files whose schema is defined by this
project — manifests, artifacts, submission payloads. Do not use raw `json.loads()`/`json.dumps()`
with manual dict validation for these files. Artifact models live in
`england_pbv.artifacts`; reuse them instead of defining parallel shapes.

JSON Schema files under `schemas/` are generated from the models by
`england_pbv.artifacts.schema_export`. Never handwrite or hand-edit a schema for a model.

#### Why:

* **Validation**: Pydantic validates types and constraints automatically
* **Type safety**: IDE autocomplete and mypy work with model fields
* **Single source of truth**: The model definition *is* the schema
* **Error messages**: Clear, structured validation errors with field paths
* **Performance**: `model_validate_json()` parses and validates in one Rust-level pass

#### Do:

```python
from pathlib import Path
from pydantic import BaseModel, ConfigDict


class CalibrationProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tokens_per_utterance: float
    seconds_per_utterance: float
    profile_version: str


# Reading
profile = CalibrationProfile.model_validate_json(
    Path("calibration.json").read_text(encoding="utf-8"),
)

# Writing — go through the atomic-write helpers for artifacts
from england_pbv.artifacts.io import write_model

write_model(path=profile_path, model=profile)
```

#### Don't:

```python
import json

data = json.loads(Path("calibration.json").read_text())
# Manual validation scattered across the codebase
if not isinstance(data.get("profile_version"), str):
    return None
```

#### Exceptions: when raw `json.loads` is correct

Use raw `json.loads()` in these cases:

* **Verifiers** that must read potentially malformed files and report multiple individual
  diagnostics. Pydantic raises one `ValidationError` on the first problem; verifiers need to
  check every field independently and report all violations at once.
* **External input** whose schema you do not control (e.g., a coding agent's own session files
  read by a discovery adapter). A strict Pydantic model would break when the external tool adds
  or changes fields, unless you use `extra="ignore"` which defeats the purpose.

```python
# Verifier: read raw, validate each field, report all problems
data: object = json.loads(raw)
if not isinstance(data, dict):
    diagnostics.append(Diagnostic.from_code(code="ARTIFACT_NOT_OBJECT"))
    return diagnostics
if "run_id" not in data:
    diagnostics.append(Diagnostic.from_code(code="ARTIFACT_FIELD_MISSING", detail="run_id"))
if "status" not in data:
    diagnostics.append(Diagnostic.from_code(code="ARTIFACT_FIELD_MISSING", detail="status"))
# ... check all fields, report all problems

# External tool file: defensive .get() with fallbacks, never crash discovery
session: object = json.loads(session_file_bytes)
if isinstance(session, dict):
    version = session.get("version")
```

* * *

### Use `model_validate_json()` instead of `model_validate(json.loads())`

Prefer `model_validate_json(raw_str)` over `model_validate(json.loads(raw_str))`.

#### Why:

`model_validate_json()` avoids constructing intermediate Python dicts, making it faster and more
memory-efficient.

#### Do:

```python
profile = CalibrationProfile.model_validate_json(raw_json)
```

#### Don't:

```python
profile = CalibrationProfile.model_validate(json.loads(raw_json))
```

* * *

### Pydantic at the edges, dataclasses inside

Use Pydantic `BaseModel` at IO boundaries (JSON artifacts, submission payloads, external data).
Use stdlib `@dataclass(frozen=True, slots=True)` for internal data passed between functions.

#### Why:

* Pydantic models are ~6-7x slower to instantiate than dataclasses
* Internal data is already validated — re-validating wastes cycles
* Dataclasses have zero external dependencies

#### Do:

```python
# Edge: an artifact read from and written to disk
class InstanceInventorySummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    adapter_id: str
    instance_count: int


# Internal: passing between functions during one pipeline stage
@dataclass(frozen=True, slots=True)
class InventoryDelta:
    adapter_id: str
    added: int
    removed: int
```

* * *

### Use `TypeAdapter` for validating lists and non-model types

When reading a JSON array or a non-model type, use `TypeAdapter`. **Always instantiate
`TypeAdapter` at module level** — instantiation builds a validator from scratch and is expensive.

For JSONL artifacts, prefer the helpers in `england_pbv.artifacts.io`, which validate one
model per line; use `TypeAdapter` for whole-array JSON files.

#### Why:

`BaseModel.model_validate_json()` expects a JSON object at the top level. `TypeAdapter` handles
any type including `list[Model]`.

#### Do:

```python
from pydantic import TypeAdapter

# Module level — instantiate once
_SPAN_LIST_ADAPTER: TypeAdapter[list[EvidenceSpan]] = TypeAdapter(
    list[EvidenceSpan],
)


def load_spans(*, file_path: Path) -> list[EvidenceSpan]:
    return _SPAN_LIST_ADAPTER.validate_json(
        file_path.read_bytes(),
    )
```

#### Don't:

```python
def load_spans(*, file_path: Path) -> list[EvidenceSpan]:
    # Expensive: builds a new validator every call
    adapter = TypeAdapter(list[EvidenceSpan])
    return adapter.validate_json(file_path.read_bytes())
```

* * *

### Use `ConfigDict` for model configuration

Configure models using `model_config = ConfigDict(...)`. Every artifact model in this project
**must** set `extra="forbid"` to catch typos in JSON keys. Prefer `frozen=True` for immutable
models.

#### Do:

```python
from pydantic import BaseModel, ConfigDict


class EvidenceSpan(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    start: int
    end: int
    label: str
```

* * *

### Keep Pydantic models as pure data structures

Keep Pydantic models as simple data containers. Put conversion/transformation logic in separate
standalone functions.

#### Why:

* **Separation of concerns**: Pydantic handles validation, conversion lives elsewhere
* **Testability**: Standalone functions easier to test
* **Reusability**: Conversion functions work without instantiating models
* **Clarity**: Distinction between "what the data is" vs "what we do with it"

#### Do:

```python
class RawSessionLine(BaseModel):
    role: str | None = None
    modality_hint: str | None = None


def _parse_modality(value: str | None) -> Modality | None:
    if value is None or value == "":
        return None
    try:
        return Modality(value)
    except ValueError:
        return MODALITY_ALIAS_MAP.get(value)


# Usage
parsed = _ParsedLine(
    modality=_parse_modality(line.modality_hint),
)
```

#### Don't:

```python
class RawSessionLine(BaseModel):
    modality_hint: str | None = None

    def get_modality(self) -> Modality | None:
        # Conversion logic mixed with data model
        ...


# Usage
parsed = _ParsedLine(
    modality=line.get_modality(),
)
```

* * *

### Use Pydantic v2 API, never v1

Always use the v2 methods. The v1 API is deprecated.

| v1 (deprecated) | v2 (use this) |
| --- | --- |
| `parse_obj(data)` | `model_validate(data)` |
| `parse_raw(json_str)` | `model_validate_json(json_str)` |
| `.dict()` | `.model_dump()` |
| `.json()` | `.model_dump_json()` |
| `schema()` | `model_json_schema()` |
| `construct()` | `model_construct()` |

* * *

## File IO and Artifact Writing

### Write artifacts through the shared IO helpers

All artifact reads and writes go through `england_pbv.artifacts.io`: `write_model`,
`read_model`, the JSONL helpers, and the `atomic_write_*` functions. Do not open artifact files
with bare `open()` or `Path.write_text()` in pipeline code.

#### Why:

* **Atomicity**: A crash mid-write must never leave a truncated artifact that a later stage
  half-reads. The helpers write to a temporary file and rename.
* **Permissions**: Private directories are created with `0700`/`0600` on POSIX via
  `ensure_private_dir`; ad-hoc writes would silently skip this.
* **Consistency**: Encoding, newline handling, and canonical JSON form are decided once.

#### Do:

```python
from england_pbv.artifacts.io import ensure_private_dir, write_model

ensure_private_dir(run_dir)
write_model(path=run_dir / MANIFEST_FILENAME, model=manifest)
```

#### Don't:

```python
run_dir.mkdir(parents=True, exist_ok=True)  # World-readable on POSIX
(run_dir / "run-manifest.json").write_text(manifest.model_dump_json())  # Not atomic
```

* * *

### Process JSONL streams line by line

Read and write JSONL artifacts one record at a time using the JSONL helpers. Do not load a whole
JSONL file into a single string or accumulate the full record list when a generator will do.

#### Why:

Transcript-derived artifacts can be large. Streaming keeps memory bounded and lets verifiers
report the line number of a broken record.

#### Do:

```python
from england_pbv.artifacts.io import read_jsonl_models

for utterance in read_jsonl_models(path=utterances_path, model_cls=NormalizedUtterance):
    process(utterance=utterance)
```

#### Don't:

```python
records = [
    NormalizedUtterance.model_validate_json(line)
    for line in utterances_path.read_text().splitlines()  # Whole file in memory
]
```

* * *

### Never emit source text from discovery or scripts

Deterministic code that touches user transcripts (discovery adapters, snapshot code, inventory
scripts) must never print, log, raise, or return the source text. Agent-facing output from
discovery is an `InstanceInventorySummary` — counts, IDs, hashes, and timestamps only.

#### Why:

The whole product promise is that raw text stays on the user's machine and never appears in
logs, exceptions, diagnostics, or agent context by accident. One stray `print()` in a debug
session can leak a private transcript.

#### Do:

```python
logger.info(
    "normalized %d utterances for adapter %s",
    len(utterances),
    adapter_id,
)
```

#### Don't:

```python
logger.info("normalized utterance: %s", utterance.text)  # Leaks source text
raise ValueError(f"bad line: {raw_line}")  # Leaks source text into tracebacks
```

* * *

## Resource Management

### Guarantee resource cleanup with try/finally for temporary files

If exceptions occur, temporary files may not be cleaned up, leading to disk space leaks — and in
this project, possibly to stray copies of snapshot data.

#### Why:

Using try/finally ensures cleanup happens regardless of success or failure.

#### Do:

```python
temp_files: list[Path] = []

try:
    # Create and process temp files
    temp_files = create_temp_files()
    process_files(temp_files)
    return results
finally:
    # Guaranteed cleanup
    for temp_file in temp_files:
        if temp_file.exists():
            temp_file.unlink()
```

#### Don't:

```python
# Cleanup only at the end - won't run if exception occurs
temp_files = create_temp_files()
process_files(temp_files)

for temp_file in temp_files:
    temp_file.unlink()
```

**Note**: Prefer a context manager (`with tempfile.TemporaryDirectory() as tmp:`) when the
resource supports one; try/finally is the fallback for multi-resource cleanup.

* * *

## Imports

### Avoid `if TYPE_CHECKING:` blocks unless absolutely necessary

Import modules directly at the top level rather than conditionally importing them for type
checking only.

#### Why:

1. **Simplicity**: Reduces cognitive load and import complexity
2. **Runtime Safety**: Ensures imported types are available at runtime if needed
3. **Consistency**: Makes all imports visible in one place
4. **Circular Import Detection**: Forces you to fix circular dependencies properly

#### Do:

```python
from england_pbv.artifacts.models import (
    NormalizedUtterance,
    SafeMistakeRecord,
)
```

#### Don't:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from england_pbv.artifacts.models import (
        NormalizedUtterance,
        SafeMistakeRecord,
    )
```

**Exception**: Only use `TYPE_CHECKING` blocks when there is a genuine circular import that
cannot be resolved through refactoring.

* * *

### Use absolute imports

Never use relative imports. Always use absolute imports rooted at the `england_pbv`
package.

#### Why:

1. Deep relative imports (like `from ...enums import Modality`) are confusing
2. Some tooling doesn't work well when mixing relative and absolute imports
3. Consistency across the codebase

#### Do:

```python
from england_pbv.artifacts.enums import Modality
from england_pbv.diagnostics.codes import Diagnostic
```

#### Don't:

```python
from ..artifacts.enums import Modality
from .codes import Diagnostic
```

#### Don't (sys.path hacks):

```python
import sys

sys.path.insert(0, str(Path(__file__).parent))
from codes import Diagnostic  # Fragile local import
```

The package lives under `src/`, so `sys.path` manipulation is never needed: `uv run` installs
the project in the environment, and tests import `england_pbv` like any other package.

* * *

### Use direct imports with standard exceptions

Prefer direct imports (`from lib import f; f()`) except for modules that are conventionally
imported as a namespace.

**Standard exceptions in this project**:

* `from england_pbv import paths` — path constructors read better as `paths.run_dir(...)`

#### Why:

Missing module members throw errors during import, not later when accessed.

#### Do:

```python
from england_pbv.artifacts.hashing import sha256_hex

sha256_hex(...)
```

#### Don't:

```python
from england_pbv.artifacts import hashing

hashing.sha256_hex(...)
```

* * *

## Documentation

### Avoid docstrings for simple dataclasses and helper functions

Don't add docstrings when the name and type hints make the purpose obvious. Only add docstrings
for non-obvious information.

#### Why:

1. **Noise Reduction**: Obvious documentation clutters code
2. **Maintenance Burden**: Docstrings need to stay in sync
3. **Type Hints Are Better**: For simple cases, type hints are self-documenting
4. **Focus on Non-Obvious**: Save documentation effort for complex logic

#### Do:

```python
@dataclass(frozen=True, slots=True)
class StageCounts:
    total_utterances: int
    kept_utterances: int
    dropped_duplicates: int


def _count_lines(file_path: Path) -> int:
    with open(file=file_path, encoding="utf-8") as f:
        return sum(1 for _ in f)
```

#### Don't:

```python
@dataclass(frozen=True, slots=True)
class StageCounts:
    """Counts for a pipeline stage."""

    total_utterances: int  # Total number of utterances
    kept_utterances: int  # Number of kept utterances
    dropped_duplicates: int  # Number of dropped duplicates


def _count_lines(file_path: Path) -> int:
    """Counts the number of lines in a file."""
    with open(file=file_path, encoding="utf-8") as f:
        return sum(1 for _ in f)
```

* * *

### Comments state constraints, not narration

Write a comment only when it says something the code cannot: an invariant, a privacy or ordering
constraint, a reason for a non-obvious choice. Never narrate what the next line does and never
keep historical notes ("used to be X").

#### Do:

```python
# Sorted before hashing: canonical hash must not depend on discovery order.
instance_ids.sort()
```

#### Don't:

```python
# Sort the instance ids
instance_ids.sort()

# Note: we previously hashed unsorted ids (changed 2025-11)
```

* * *

## Command Line Tools

### Expose scripts as `python -m` modules

Every deterministic script is a module under `england_pbv` with a `main()` entry point,
run as `uv run python -m england_pbv.<module>`. No standalone scripts outside the
package, and no business logic in shell.

### Use argparse for command line arguments

Use argparse for parsing command line arguments. Define the parser in `main()`, keep the logic in
plain functions that take typed parameters, and cover the logic with tests that bypass argparse.

### Keep terminal output text-safe

CLI output follows the same privacy rule as logging: counts, IDs, paths, and diagnostics only —
never transcript text. Long-running operations should report progress through the shared
`progress` module rather than ad-hoc prints.

* * *

## Formatting Requirements

### Let ruff decide, then keep its output

`uv run ruff format .` is the single formatter. Never hand-format against it, never disable it
for a region without a stated reason.

### Maximum line length: 100 characters

Keep all lines under 100 characters (enforced by ruff).

### PEP8 compliance

* No spaces in empty lines
* No trailing spaces
* Two empty lines between top-level functions
* One empty line at the end of the file

### Use Path instead of string paths

Use `pathlib.Path` for all file path operations, not string manipulation.

#### Do:

```python
from pathlib import Path

report_path = run_dir / "reports" / "verification.json"
```

#### Don't:

```python
import os

report_path = os.path.join(run_dir, "reports", "verification.json")
```

* * *

## Python Version

Use Python 3.12+ syntax:

* `dict`, `list`, `tuple` (not `Dict`, `List`, `Tuple`)
* `X | None` (not `Optional[X]`)
* `type AdapterId = str` (not `AdapterId: TypeAlias = str`)
* PEP 695 style generics where applicable

Ruff's `UP` rules enforce most of this automatically.

* * *

## Testing Conventions

* Tests live under `tests/`, named `test_<area>_<topic>.py`, with globally unique file names and
  no `__init__.py` files.
* Use `tmp_path` for every filesystem test. Never write into the repository tree or the real
  runtime root.
* Fixture data lives under `fixtures/<adapter_id>/<variant>/` with a `fixture.json` metadata
  file. Every adapter needs success, empty, malformed, unsupported-schema, and migration
  fixtures.
* All fixture content is synthetic. No real user text, no real names, no real paths containing a
  user name. Secret-looking values must be unmistakably fake (e.g. `sk-FAKEFAKEFAKE0000`).
* Assert on diagnostics by code, not by message text.

#### Do:

```python
def test_normalization_skips_empty_transcript(tmp_path: Path) -> None:
    transcript = tmp_path / "session.jsonl"
    transcript.write_text("", encoding="utf-8")

    result = normalize_transcript(
        transcript_path=transcript,
        tokenizer_version="1.0.0",
        min_word_count=3,
    )

    assert len(result) == 0, "empty transcript yields no utterances"
```

#### Don't:

```python
def test_normalization() -> None:
    # Reads a file inside the repository and depends on developer machine state
    result = normalize_transcript(
        transcript_path=Path("fixtures/session.jsonl"),
        tokenizer_version="1.0.0",
        min_word_count=3,
    )
    assert result
```

* * *

## Checklist for New Modules

When starting a new pipeline module:

1. Take all runtime locations from `england_pbv.paths`; add new layout there if needed
2. Reuse enums from `artifacts/enums.py` and models from `artifacts/models.py`; extend them in
   their home modules instead of redefining
3. Expose `PRODUCER_VERSION: str` if the module writes an artifact
4. Write analysis logic as pure functions in a separate module (no IO mixed in)
5. Create a `main()` orchestrator: load -> compute -> save, with all writes going through
   `artifacts/io`
6. Use frozen dataclasses for all internal result containers
7. Report every failure as a registered `Diagnostic`, never a bare string
8. Add assertions after every load and merge
9. Add tests with synthetic fixtures for success, empty, and malformed inputs
10. Run the full quality gate before finishing

* * *

## Summary

The most commonly overlooked rules are:

1. **Always use dataclasses instead of tuples**
2. **Always use keyword arguments for 2+ heterogeneous parameters**
3. **Never return tuples from functions**
4. **Take all runtime paths from `england_pbv.paths`**
5. **Use constants and enums for all magic strings**
6. **Use `None` for missing data, never `0.0` or `""`**
7. **Use absolute imports, never relative imports**
8. **Write artifacts only through `artifacts/io` (atomic, private-permission writes)**
9. **Never print, log, or return source text from deterministic code**

Follow these rules consistently, and keep the `uv run` quality gate green.
