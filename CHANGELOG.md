# Changelog

## 0.4.0 — 2026-04-19

### Changed
- **Translation prompts split the "code preservation" rule so comments
  and user-facing string literals inside code blocks are translated
  by default, and source-language labels inside inline code spans are
  translated too.** The five instruction blocks (`TRANSLATE_INSTRUCTION`,
  `BATCH_TRANSLATE_INSTRUCTION`, `REFINE_INSTRUCTION`,
  `BATCH_MULTI_TRANSLATE_INSTRUCTION`, `BATCH_REFINE_INSTRUCTION`) now
  tell the model:
  1. **Inside fenced code blocks:** preserve code syntax, identifiers,
     function and variable names, module references, API paths, type
     names. *But you MUST translate comments and user-facing string
     literals (messages shown to end users) inside those code blocks
     — by default, not only on demand.* This is the strengthening
     over pre-v0.4 behaviour: the old rule permitted comment
     translation but most models skipped it to be safe, so
     `print("안녕")` stayed in Korean in English builds. It now
     becomes `print("hello")`.
  2. **Inside inline code spans:** preserve identifiers, file paths,
     URLs, shell commands, code-literal content. *Exception: when an
     inline code span contains prose written in the source language
     rather than the target-language character set (for example
     `` `게임코드` `` in a Korean → English translation), translate it
     naturally — the author used code styling for a human label, not
     an identifier.* This fixes the most common complaint: target
     documents still containing source-language text because the
     author put Korean labels in backticks.
  3. **For critical identifier mappings** (`getUserInfo`,
     `/api/v1/users`, product names that must never shift), add a
     glossary entry with `--glossary-mode placeholder` so the term
     is hard-protected via sentinel substitution and round-trip
     verified. This is the only fully-deterministic path for
     cross-block identifier stability; the prompt on its own is
     guidance the model can override.
  4. Bare URLs and file paths outside code contexts are still
     preserved (reintroduced in v0.4 after an initial sweep removed
     them — nothing downstream rewrites URLs, so they must stay put
     unless a glossary entry directs otherwise).
  5. Anchor IDs (`{#section}`) and the HTML attribute allowlist
     (`href`, `id`, `class`, …) remain hard-protected in
     `placeholder.py`, unchanged, so internal links do not break.
  6. Format-string interpolation tokens (`{{name}}`, `%s`, `${var}`)
     still have their own preservation rule — they are format
     syntax, not identifiers, and the glossary cannot replace them.

- **Default `glossary_mode` flipped from `"instruction"` to
  `"placeholder"`** in both the `MdpoLLM` constructor and the
  `translate` / `translate-multi` CLI commands. Placeholder mode
  substitutes each glossary term with an opaque `⟦P:N⟧` token before
  the LLM call and restores it after, so the model cannot mistranslate
  or mangle protected terms. Pass `glossary_mode="instruction"`
  (kwarg) or `--glossary-mode instruction` (CLI) to keep the old
  prompt-block behaviour — useful for glossary terms that start/end
  with non-word characters (`.NET`, `C++`) which the placeholder
  regex cannot match.

### Added
- **Per-directory glossary cascade (`translate-dir`)**: `process_directory`
  auto-discovers a per-file glossary chain so different subtrees can
  treat the same term differently without re-instantiating the
  processor. For every source file the resolver walks from `source_dir`
  down to the file's directory, layering each `glossary.json` it finds,
  then applies `./glossary.json` from the current working directory,
  then (topmost) the `--glossary PATH` CLI override. A
  `"__remove__"` sentinel value unsets an inherited term (child opts
  out of the parent mapping); any other value (`null` / string /
  per-locale dict) follows existing semantics (do-not-translate /
  force specific translation / locale-specific form). Results are
  cached per directory so sibling files reuse the merged parent
  chain, and per-effective-glossary so the placeholder registry's
  compiled patterns are shared across files with identical mappings.
  No new CLI flag — auto-detection runs whenever `--glossary` is not
  passed; passing `--glossary PATH` keeps its single-file-override
  semantics and slots in as the topmost cascade level. Under `-v` a
  one-line INFO log per file lists the resolved chain for debugging.
- **`translate-dir --translate-paths` (opt-in)**: translate filesystem
  path segments (directory names and markdown file stems) so the target
  tree uses localized filenames. Segments are tracked in a dedicated
  `_paths.po` catalog that lives under `--po-dir` (or the target
  directory when `--po-dir` is omitted), with one PO entry per distinct
  source segment keyed by `msgctxt="path::segment::<raw>"`. Translations
  flow through the same `_call_llm` pipeline as content blocks, so
  caching, glossary configuration, and `usage` accounting reuse the
  existing machinery. Output segments are sanitized via
  `parser.slugify_path_segment` (NFC-normalised, whitespace collapsed,
  Windows / POSIX filesystem-reserved characters stripped, leading `.`
  refused) and per-directory `-2` / `-3` disambiguators resolve slug
  collisions deterministically. File extensions are preserved verbatim;
  dotfile segments (`.github`, `.well-known`) pass through unchanged. A
  `path_map.json` with the full source → target relative-path mapping
  is written alongside the translated tree so downstream tooling (link
  rewriters, sitemaps, CI) can resolve the pairing without re-running
  the translator. Link text and URLs inside translated Markdown are
  deliberately NOT rewritten — that is a separate cross-reference
  problem, and solving it in this task would invalidate every
  translated document's internal anchors. PO paths remain keyed on the
  SOURCE relative path so incremental re-runs still hit their
  previously-processed entries even when output filenames move. Tokens
  billed by path-segment translation fold into the directory-level
  `Receipt` so operators see the real cost of a `--translate-paths` run.
- **`processor.py` — `process_document_multi` / multi-target translation
  (experimental)**: new method on `MarkdownProcessor` that translates a
  single Markdown source into multiple target languages in a single
  batched LLM call per source group. Source-side decomposition
  (placeholder substitution, glossary matching, reference lookup) runs
  ONCE per block regardless of `len(target_langs)`, so the input-token
  bill is amortised across every target while only output tokens grow
  with the fan-out. Each language has its own per-lang PO file and
  per-lang reference pool seeded from that PO; translations do not
  cross languages. Per-lang fallback kicks in when the model returns
  partial per-lang coverage — any lang present commits directly, and
  any lang absent for a given block drops into an independent
  single-target per-entry call, so PO files are never left
  half-populated. Rejects `mode="refine"` (same-language by contract)
  and does not support `inplace=True`. Returns a dict with `by_lang`
  (per-lang `ProcessResult`, `receipt=None`), a top-level shared
  `Receipt` (billed once across every lang; `target_lang` names the
  comma-joined locale list for auditability), `source_path`, and
  `target_langs`. Distinctness guards reject duplicate target/PO
  paths across langs and reject any alias with the source path up
  front.
- **`batch.py` — `MultiTargetBatchTranslator`**: new class that wraps
  `BatchTranslator`'s partitioning / bisection logic but validates
  per-block responses as `{lang: translation}` dicts. Partial per-lang
  coverage is preserved — any lang that came back with a well-typed
  string is kept, and blocks whose value is not a dict (or a dict with
  no valid per-lang strings) drop into bisection. Deduplicates /
  validates `target_langs` up front; empty / non-string entries raise.
- **`prompts.py` — `BATCH_MULTI_TRANSLATE_SYSTEM_TEMPLATE` /
  `BATCH_MULTI_TRANSLATE_INSTRUCTION`**: new prompt templates for the
  multi-target wire. Input is the usual `{key: source}` JSON; output
  is `{key: {lang: translation}}` with every key present exactly
  once, every per-block value containing exactly the target locales,
  and the same placeholder / code-block / interpolation-token rules
  as the single-target batch prompt.
- **CLI — `translate-multi` subcommand**: `mdpo-llm translate-multi
  SOURCE --target-template '{lang}/README.md' --langs ko,ja,zh-CN
  --model gpt-4o` drives the multi-target path with the standard
  batching / glossary / validation / receipt flags reused from
  `translate`. `--target-template` and `--po-template` (optional)
  both require the literal substring `{lang}` so the per-lang path
  is explicit; the CLI fails up front with a usage error otherwise.
  `--json-receipt` dumps the shared receipt.
- **`__init__.py`**: `BatchTranslator` and `MultiTargetBatchTranslator`
  exported at the package root for callers that wire their own
  message construction.
- **Tests** — `tests/test_batch.py::TestMultiTargetBatchTranslator`
  covers happy-path fan-out, empty input, empty / non-string
  `target_langs` rejection, dedup + order preservation, partial
  coverage (kept as-is), non-dict value (bisected), malformed JSON
  (bisected), single-entry failure returning `{}`, partitioning by
  entry count, and extra langs in the response being dropped.
  `tests/test_batched_processing.py::TestMultiTargetProcessing`
  covers the single-call-produces-all-langs invariant, default PO
  path derivation per lang, refine-mode rejection, empty /
  duplicate / colliding path rejection, source-path aliasing,
  duplicate lang dedup, the per-lang fallback path when the
  multi-response is partial, the re-run-no-op path after a clean
  first pass, and the billed-once receipt (one `api_calls`, locales
  joined in `target_lang`).

  The human-runnable "canonical-seeded" baseline is already
  expressible via the existing single-target `process_document`
  repeated per lang — no new code is required to compare the two
  approaches after merge.

- **`processor.py` — `batch_concurrency` kwarg / `--batch-concurrency N`
  CLI flag (experimental)**: `MarkdownProcessor` accepts a new
  `batch_concurrency: int = 1` kwarg that lets multiple section-aware
  batches from a single document fly in parallel. The first group is
  always processed on the calling thread so the reference pool is
  seeded before any worker reads from it; remaining groups are
  submitted to a `ThreadPoolExecutor` of size `batch_concurrency`.
  Worker-local stats and `_UsageAccumulator` instances are merged
  into the shared receipt under a single lock so the user-visible
  output (PO file, `BatchStats`, `Receipt`) matches the sequential
  path. Defaults to `1` (off) — callers must opt in. Non-positive /
  non-int values clamp to `1` so a typo degrades to the safe path
  rather than raising deep inside the executor. Ignored when
  `batch_size == 0` (sequential path) and when a document partitions
  into a single group. Forwarded through
  `_sibling_refine_processor` so `translate --refine-first` honours
  the same concurrency budget across both passes.
- **CLI — `--batch-concurrency N`**: forwarded via `_add_shared_flags`
  so every batch-capable subcommand (`translate`, `translate-dir`,
  `refine`, `refine-dir`, `estimate`) accepts the new flag.
- **Tests** — `tests/test_batched_processing.py::TestBatchConcurrency`
  covers the default (`batch_concurrency == 1` leaves the executor
  untouched), the clamp on non-positive / non-int inputs, the
  seed-first invariant (first LLM call runs alone before any worker
  starts), genuine overlap on the parallel path, aggregation of
  `batched_calls` / receipt tokens across workers, and the
  single-group fall-through.
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
  same reason.
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
