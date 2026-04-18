# Changelog

## Unreleased

### Added
- **`processor.py` — `mode` kwarg / `refine` path**: `MarkdownProcessor`
  gains a `mode: Literal["translate", "refine"]` constructor kwarg
  (default `"translate"`). Refine mode polishes Markdown in its **source
  language** without translating or switching languages, reusing the
  entire translate pipeline (parser, PO tracking, batching, reference
  pool, placeholders) and swapping only the prompt template and
  validator purpose. `refined_path` / `refined_dir` kwargs on
  `process_document` / `process_directory` route the refined output to a
  separate location; the PO `msgid` is NEVER overwritten — refined text
  lands in `msgstr`. `inplace=True` combined with `mode="refine"` raises
  (the contract forbids msgid overwrite).
- **`processor.py` — `translate --refine-first` composition**: new
  `refine_first=True` + `refined_path` kwargs on `process_document`
  (and `refined_dir` on `process_directory`) run a refine pass before
  translating. Tokens from both passes flow into the returned
  `Receipt`. `refine_lang` picks the refine-pass language (defaults to
  `target_lang` only when source==target).
- **`prompts.py` — `REFINE_SYSTEM_TEMPLATE` / `REFINE_INSTRUCTION` /
  `BATCH_REFINE_SYSTEM_TEMPLATE` / `BATCH_REFINE_INSTRUCTION`**: new
  prompt templates for same-language polish. Both variants explicitly
  forbid language switching, preserve Markdown structure, keep
  `⟦P:N⟧` placeholders intact, and ban invention of new content.
- **`validator.py` — `purpose` kwarg + `language_stability` check**:
  `validate()` accepts `purpose: Literal["translate", "refine"]`
  (default `"translate"`). Refine purpose drops the
  `target_language` presence check (source language == target language
  makes it meaningless) and adds `language_stability`, which fails
  whenever `detect_languages(source) != detect_languages(output)` —
  catching a model that silently switches languages mid-refinement.
- **CLI — `refine` / `refine-dir` subcommands**: new same-language
  polish entry points mirroring `translate` / `translate-dir` flags.
  `translate` / `translate-dir` gain `--refine-first`,
  `--refined-path` / `--refined-dir`, and `--refine-lang` for the
  compose path. `--inplace` help text calls out deprecation and
  points users at the refine alternative.
- **Tests** — `tests/test_validator.py` covers the refine-purpose
  branches (target-language check suppressed, `language_stability`
  flagged on language drift, structural checks still run);
  `tests/test_batched_processing.py::TestRefineMode` covers prompt
  selection, msgid preservation, the `inplace`/`refine_first`
  validation guards, and the compose receipt aggregation.

### Deprecated
- **`inplace=True` on `MarkdownProcessor.process_document` /
  `process_directory`**: emits a `DeprecationWarning` pointing at refine
  mode and `refined_path` / `translate --refine-first`. The flag is
  slated for removal in v0.5. Callers who intentionally rewrote source
  content after translating should switch to `mode="refine"` with a
  separate `refined_path` — the refine contract captures the "polished
  document" intent without clobbering the original.

### Fixed
- **`processor.py` — untranslated-warning false positives**: the
  "LLM returned untranslated output" check fired on every code block
  because rule 3 of the translation instruction tells the model to keep
  code as-is, so `output == source` is the expected outcome. A single
  real-world v0.3 run produced 34 spurious warnings and buried genuine
  prose regressions. The warning now skips entries whose block type is
  `code` in both the sequential and batched paths; non-code block types
  still warn as before.

### Changed
- **`prompts.py` — `BATCH_TRANSLATE_INSTRUCTION` rule 1**: tightened from
  "No prose, no explanations, no Markdown code fences" to an explicit
  "response MUST start with `{` and end with `}`; no ```json fences, no
  surrounding backticks, no language tag, no preamble or epilogue".
  Models occasionally wrapped their JSON reply in fences under the old
  phrasing, which triggered the batch parser's bisection fallback; the
  sharper wording removes the ambiguity.

### Added
- **`placeholder.py` — T-6 built-in patterns (`anchor`, `html_attr`)**:
  every `MarkdownProcessor` now registers two always-on placeholder
  patterns on top of the T-4 machinery:
  - `anchor` protects Kramdown / Pandoc anchor IDs — the plain
    `{#overview}` form AND longer IAL variants that co-declare classes
    or key=val attrs (`{#overview .lead}`, `{#detail key=val}`).
    Mangling an anchor breaks every `#overview` link pointing at the
    heading.
  - `html_attr` protects raw-HTML attribute pairs on an allowlist of
    identity / link / structural attributes (`class`, `id`, `href`,
    `src`, `srcset`, `rel`, `target`, `name`, `for`, `style`, `type`,
    `role`, `lang`, `dir`, `xml:lang`, `xmlns`, `action`, `method`,
    `width`, `height`, `data-*`). Matches double-quoted, single-quoted,
    AND HTML5 unquoted values (`<img width=320 height=240>`) and is
    compiled case-insensitively so uppercase attribute names
    (`<A HREF="/docs" Class="bare">`) are protected too. Tag boundary
    detection (`HTML_TAG_OPEN_RE`) is quote-aware — attributes that
    follow a quoted value containing `>` (e.g.
    `<a title="1 > 0" href="/docs">`) are still recognised as in-tag.
    Real-world runs showed these link-attribute strings getting
    translated or reordered. Attribute values that normally contain
    user-facing prose — `alt`, `title`, `aria-label`, `placeholder`,
    `label` — are deliberately NOT on the allowlist so the LLM still
    gets to translate accessibility and tooltip text.

  Built-ins additionally skip Markdown code contexts entirely —
  fenced blocks (```` ``` ```` and `~~~`, including blockquoted and
  nested-blockquoted fences like `> ~~~html` / `>> ~~~html`),
  indented code blocks (4-space / tab indent after a blank line or
  directly after a heading, with list-continuation lines excluded),
  and backtick inline code spans (any run length, spanning newlines
  per CommonMark). Partial tokenization of a literal HTML / anchor
  example would otherwise let the model rewrite the example while
  the token count still balanced.

  There is no opt-out flag — the brief explicitly forbids one. Missing
  any of these tokens in the LLM output is a structural fail via the
  placeholder round-trip check (same hard-fail path used by every other
  pattern), independent of `validation` mode. A user-supplied
  `placeholders` kwarg layers additional patterns on top of the
  built-ins without disabling them — a same-name user pattern only
  suppresses the default when the caller explicitly sets
  `replace_builtin=True` on the registration (the no-opt-out contract
  stays intact by default). Glossary placeholder mode layers its own
  patterns last — the four-layer composition is consolidated in
  `MarkdownProcessor._build_effective_registry`.
- **`placeholder.py` — `check_structural_position`**: token-count
  round-trip alone could not catch a model that kept `⟦P:N⟧` exactly
  once but relocated it. The new check compares structural placement
  between source and decoded translation:
  - Heading anchors use per-heading `(anchor, heading_ordinal)`
    multisets, so a swap between two headings still fires even when
    total anchor count is unchanged. Heading detection covers both
    ATX (`#`) and setext (`===` / `---` underlined) styles.
  - HTML attributes use a multiset of per-tag attribute signatures
    (see `_attr_tag_signatures`) rather than tag ordinals, so legal
    cross-language inline-tag reorders
    (`<a href="/a">A</a> and <a href="/b">B</a>` →
    `<a href="/b">B</a>와 <a href="/a">A</a>`) aren't flagged while
    attribute mixes across tag boundaries still are.
  Reports a `placeholder_position` issue on any drift. Integrated
  into `MarkdownProcessor._apply_validation` alongside
  `check_round_trip` so position drift is a structural fail
  independent of `validation` mode.
- **`placeholder.py` — `PlaceholderPattern.predicate`**: optional
  `Callable[[text, start, end], bool]` attached to a registered
  pattern; when supplied, `PlaceholderRegistry.encode` filters each
  candidate match through it before treating the match as a
  substitution. Lets a pattern restrict itself to a surrounding
  context a vanilla regex can't express. The T-6 `html_attr`
  built-in uses this via `_is_inside_html_tag` (which in turn calls
  `_in_quoted_value`) so only attribute pairs that are BOTH inside a
  `<...>` opening tag AND at the top level of that tag — not nested
  inside another attribute's quoted value — are substituted.
  Attribute-like substrings in prose (`` `class="primary"` `` inside
  backticks, `href=/docs` in a sentence) and inside translatable
  attribute values (`<a title='see href="/docs"' href="/real">`) are
  all left alone so the LLM can still translate the prose while the
  real attributes stay protected.
- **`processor.py` — `glossary_mode` kwarg / `--glossary-mode` CLI flag**:
  selects how glossary terms are fed to the LLM. `"instruction"`
  (default, v0.4 back-compat) keeps the existing prompt-block flow.
  `"placeholder"` substitutes each matching term with an opaque
  `⟦P:N⟧` token pre-call and restores the target-language form (or the
  original term for do-not-translate entries) post-call, reusing the
  T-4 placeholder machinery so the round-trip check automatically flags
  any dropped glossary token. Matching is case-sensitive
  word-boundary (`\bterm\b`); trailing morphology (e.g. `"APIs"`
  matching `"API"`) is deliberately NOT matched — false-negatives are
  preferred over mid-word false-positives. Terms whose first or last
  character isn't a word character (`.NET`, `C++`) are skipped for the
  same reason. v0.5 will flip the default to `"placeholder"`.
- **`placeholder.py` — `Placeholder.replacement` field**: optional
  override consumed by `PlaceholderRegistry.decode`. When set, decode
  restores the token to this string instead of `original`; the glossary
  placeholder path uses it to emit the target-language form on
  restore. Defaults to `None` so pre-existing callers keep an identity
  decode.
- **`placeholder.py` — `PlaceholderRegistry` / `PlaceholderMap` /
  `check_round_trip`**: shared-core placeholder substitution framework.
  Callers register named regex patterns; the processor replaces every
  match with an opaque `⟦P:N⟧` token before the LLM sees the block and
  restores the original content afterwards. T-4 ships the module with
  zero built-in patterns — downstream tasks (T-5 glossary do-not-translate,
  T-6 anchor preservation, T-7 refine) register their own. Pre-existing
  `⟦P:N⟧` literals found in the source (e.g. documentation explaining the
  token syntax) are recorded as identity entries with
  `pattern_name == "__literal__"`: numbering skips their indices, decode
  leaves them in place, and the round-trip check requires the LLM to
  preserve them verbatim — duplicate literals in the source expect a
  matching count in the output.
- **`processor.py` — `placeholders` kwarg**: `MarkdownProcessor` accepts
  an optional `PlaceholderRegistry`. When supplied, every source block is
  encoded before the LLM call (both sequential and batched paths) and the
  response is decoded before post-processing. Reference-pool lookups and
  glossary matching still operate on the human-readable source text so
  similarity and term detection stay effective. `None` (default) keeps
  the pre-T-4 behaviour exactly.
- **`prompts.py`**: both `TRANSLATE_INSTRUCTION` and
  `BATCH_TRANSLATE_INSTRUCTION` gain a rule that any `⟦P:N⟧` token must
  be copied through unchanged — no translation, renumbering, removal, or
  duplication.
- **`validator.py` — `placeholder_map` / `encoded_translation` kwargs**:
  `validate()` accepts an optional `PlaceholderMap` plus the pre-decode
  LLM output. Structural checks (heading level, fence count, inline
  code count, glossary, target language) always run on the user-visible
  decoded `translation` argument — this keeps patterns that cover
  Markdown syntax (fences, inline code, headings) from falsely failing
  the checks. Only the round-trip check consumes `encoded_translation`;
  a missing, duplicated, or unexpected token is a structural fail
  independent of `validation` mode.
- **Tests** — `tests/test_placeholder.py` covers encode/decode
  round-trip, overlap resolution (earliest-start, longest-on-tie,
  nested-inner dropped, zero-width skipped), the round-trip checker's
  three failure modes, registry API surface (pre-compiled vs string
  regex, flags), validator integration, and processor wiring smoke
  tests.

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
