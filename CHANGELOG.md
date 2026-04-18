# Changelog

## Unreleased

### Added
- **`processor.py` — `ProgressEvent` / `progress_callback`**: new
  `MarkdownProcessor` kwarg for a UI-agnostic progress hook. The
  callable receives a frozen `ProgressEvent(kind, path, index, total,
  status)` at `document_start` / `document_progress` / `document_end`
  (per document, per batch in batched mode or per pending entry in
  sequential mode) and `directory_start` / `file_start` / `file_end` /
  `directory_end` (for `process_directory`). Callback exceptions are
  logged and swallowed so a broken UI can't abort a translation.
  Events are emitted from worker threads in the directory path.
- **CLI — rich progress bar**: `translate` renders a batch bar for the
  document; `translate-dir` renders a file-count bar across the run.
  Auto-suppressed when stderr is not a TTY, when `-v` turns on
  structured logging, when `MDPO_NO_PROGRESS` is set, or via the new
  `--no-progress` flag. `rich>=13.0` is a new optional dependency
  (`pip install mdpo-llm[progress]`); when it isn't installed the bar
  silently disables without affecting core functionality.
- **Tests** — `tests/test_batched_processing.py::TestProgressHook`
  covers document events (batched and sequential paths), directory
  events (start/end + per-file status), the no-op re-run path, the
  exception-still-closes-document contract, and the
  callback-exception-is-swallowed contract.
- **`results.py` — `Receipt`**: per-run token/cost/duration summary with
  fields `model`, `target_lang`, `source_path`, `target_path`, `po_path`,
  `input_tokens`, `output_tokens`, `total_tokens`, `api_calls`,
  `duration_seconds`, per-1M USD prices, and per-category / total cost in
  USD. Cost fields are `None` for models not listed in `litellm.model_cost`
  and render as `"—"`. Exposed as `result.receipt` on both `ProcessResult`
  and `DirectoryResult` (backward-compatible — omitting the kwarg still
  constructs a valid result with `receipt=None`).
- **`processor.py`**: sums `response.usage.prompt_tokens` /
  `completion_tokens` across every `litellm.completion` call (batched,
  bisected, and per-entry fallback) and measures wall-clock. Pricing is
  resolved from `litellm.model_cost` with a provider-prefix fallback
  (`anthropic/claude-sonnet-4-5` → `claude-sonnet-4-5`).
  `process_directory` aggregates per-file receipts into a directory-level
  receipt whose duration is the wall-clock of the concurrent run.
- **CLI**: `mdpo-llm translate` / `translate-dir` print the receipt block
  to stderr after the JSON result. New `--json-receipt PATH` flag writes
  the structured receipt to a file for CI consumers.
- **Tests** — `tests/test_results.py` (Receipt dataclass, render, optional
  fields), `tests/test_batched_processing.py::TestReceipt` (token
  accumulation, unpriced-model fallback, directory aggregation).

## 0.3.0

### Added
- **`batch.py` — `BatchTranslator`**: partitions pending items by entries
  (default 40) and chars (default 8000), issues JSON-mode calls, parses the
  response, validates keys, and recursively bisects on any failure. A
  single-entry batch that still fails returns `{}` so the caller can fall
  back to the per-entry path.
- **`prompts.py`**: `BATCH_TRANSLATE_SYSTEM_TEMPLATE` and
  `BATCH_TRANSLATE_INSTRUCTION` with a strict "same keys, no prose, no
  fences" contract and a consistency rule for tone/register.
- **`processor.py`**:
  - `batch_size` (default 40) and `batch_max_chars` (default 8000) kwargs.
  - `_process_entries_batched` path that seeds the reference pool from
    the PO, deduplicates top-K similar pairs across the batch into a
    single reference block, unions the glossary, and hands off to
    `BatchTranslator`.  Missing keys fall back to the per-entry
    `_call_llm` path.
  - Section-aware chunking respects top-level path boundaries when
    splitting batches.
  - `validation: "off" | "conservative" | "strict"` runs post-translation
    structural checks and flags failing entries fuzzy.
  - `enable_prompt_cache=True` marks the stable system prefix with
    `cache_control: {"type": "ephemeral"}` for prompt-caching providers.
  - `estimate(source, po=None)` dry-runs cost and pending-block counts
    without any API calls.
- **`validator.py`**: conservative structural checks (heading level,
  fence count, glossary do-not-translate preservation, target-language
  presence) with `mode="strict"` adding an inline-code count check.
- **`results.py`**: frozen dataclasses (`ProcessResult`, `Coverage`,
  `BatchStats`, `DirectoryResult`) with a dict-like `__getitem__` /
  `__contains__` / `.get()` shim for backward compatibility.
- **CLI** — `python -m mdpo_llm` with `translate`, `translate-dir`,
  `estimate`, `report` subcommands.  Also exposed as the `mdpo-llm`
  console script via `project.scripts`.
- **Tests** — `tests/test_batch.py`, `tests/test_validator.py`,
  `tests/test_results.py`, `tests/test_batched_processing.py` (including
  bisection, section-aware chunking, validation gating, inplace mode with
  batching, and the estimator).

### Changed
- `MarkdownProcessor.process_document` and `.process_directory` now
  return `ProcessResult` / `DirectoryResult` dataclasses.  Existing
  dict-style consumers (`result["coverage"]`, `result.get(...)`) keep
  working via the mapping shim.
- `conftest.mock_completion` now detects JSON-object user content and
  round-trips the `[TRANSLATED] ` sentinel for both sequential and
  batched paths.
- `TestSequentialProcessing` and the partial-failure tests opt into the
  sequential path via `batch_size=0` so their per-call assertions hold.

### Migration
- No data migration required.  Existing PO files work unchanged.
- Batched mode is the default; pass `batch_size=0` to restore v0.2
  per-entry behaviour.
