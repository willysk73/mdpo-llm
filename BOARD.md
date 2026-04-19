# BOARD

Task queue for `/ccx:supervisor`. Briefs land in `.ccx/tasks/<id>.md` once
dispatched. Full rationale for each item lives in `docs/ROADMAP.md` —
this file is the **queue card**, not the spec.

## Direction

Post-v0.3 priorities, ordered by signal-per-effort:

1. **Observability before tuning.** Receipt (T-1) and progress bar (T-2) land
   first so every later change has token/cost/wall-clock numbers to compare
   against. No performance work without a baseline.
2. **Fidelity machinery.** Placeholder substitution (T-4) is the foundation
   for anchor preservation (T-6), glossary (T-5), and refine (T-7). Ship T-4
   standalone; T-5/T-6/T-7 depend on it.
3. **Stop warning noise.** Code-block false-positive warning (T-3) is a
   two-line fix that keeps real regressions visible; land anytime.
4. **Refine / translate split (T-7)** after placeholders exist so the refine
   path reuses T-4's machinery. Deprecates `--inplace` in v0.4, removes in v0.5.
5. **Defer large surface changes.** Batch concurrency (T-8), multi-target
   (T-9), filename translation (T-10) are v0.5-sized. Don't start them until
   earlier items have shipped and metrics from T-1 justify the effort.

**Release checklist.** Every task that changes public API, CLI flags, or
constructor signature MUST update `README.md` **in the same PR** — not as a
separate cleanup. `CHANGELOG.md` gets an entry under `## Unreleased` on merge.

**Scope overlap note.** `src/mdpo_llm/processor.py` is touched by nearly
every task; the supervisor's overlap gate will serialize most of the work.
That's expected — this codebase is small and sequential execution is fine.
Parallelism gains come from dependency-based dispatch (a merged T-4 unblocks
T-5/T-6/T-7 simultaneously) and supervisor-managed worktrees/briefs, not raw
concurrency.

## Tasks

```yaml
- id: T-1
  title: "Receipt dataclass + token/cost/duration tracking"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/results.py
      - src/mdpo_llm/__main__.py
      - tests/test_results.py
      - tests/test_batched_processing.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: high
  depends_on: []
  brief: .ccx/tasks/T-1.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-1
  branch: ccx/T-1
  started_at: "2026-04-18T06:41:11Z"
  finished_at: "2026-04-18T07:05:10Z"
  exit_status: approved
  notes: |
    Roadmap #2. Sum token usage returned by litellm.completion across all
    calls (batched + per-entry), resolve per-1M pricing via
    litellm.model_cost with a "—" fallback for unpriced models, measure
    wall-clock, and emit a human-readable "receipt" block at the end of
    every translate / translate-dir run. Add --json-receipt PATH for CI
    consumers. Include model, target_lang, source/target/po paths in the
    output. All of this must land on the existing ProcessResult /
    DirectoryResult dataclasses (backward-compatible dict shim preserved).

- id: T-2
  title: "Per-document progress display (CLI)"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/__main__.py
      - tests/test_batched_processing.py
      - README.md
      - CHANGELOG.md
      - pyproject.toml
    exclude: []
  status: merged
  priority: high
  depends_on: []
  brief: .ccx/tasks/T-2.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-2
  branch: ccx/T-2
  started_at: "2026-04-18T07:06:35Z"
  finished_at: "2026-04-18T07:26:30Z"
  exit_status: approved
  notes: |
    Roadmap #1. Add a progress-event hook on the processor (emit per
    batch / per file — library stays UI-agnostic) and wire it to
    rich.progress in the CLI. One bar per file in process_directory, one
    bar for batches in process_document. Suppress when non-TTY or when
    -v enables structured logging so CI logs stay clean. Pyproject gets
    a new optional dependency (rich or tqdm — prefer rich for TTY
    detection).

- id: T-3
  title: "Exempt code blocks from 'LLM returned untranslated' warning"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/prompts.py
      - tests/test_batched_processing.py
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: normal
  depends_on: []
  brief: .ccx/tasks/T-3.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-3
  branch: ccx/T-3
  started_at: "2026-04-18T08:03:41Z"
  finished_at: "2026-04-18T08:09:40Z"
  exit_status: approved
  notes: |
    Roadmap #4. The v0.3 real-world test produced 34 false-positive
    warnings because code blocks legitimately round-trip unchanged and
    the detector flags output==source. Fix by skipping the warning when
    the block type is `code`. While the prompt file is open, tighten
    the batch instruction to explicitly forbid wrapping JSON output in
    Markdown fences (still bites occasionally).

- id: T-4
  title: "Placeholder substitution framework (shared core)"
  scope:
    include:
      - src/mdpo_llm/placeholder.py
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/prompts.py
      - src/mdpo_llm/validator.py
      - tests/test_placeholder.py
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: high
  depends_on: []
  brief: .ccx/tasks/T-4.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-4
  branch: ccx/T-4
  started_at: "2026-04-18T07:28:36Z"
  finished_at: "2026-04-18T08:03:20Z"
  exit_status: approved
  notes: |
    Roadmap #5 + #6 foundation. New module `placeholder.py` that replaces
    configurable source patterns with opaque tokens before the LLM sees
    them (format e.g. `⟦P:0⟧`), restores after. Processor wires the
    pipeline pre/post-call; prompt gains a placeholder-preservation
    rule; validator gains a round-trip check (every input placeholder
    must appear exactly once in output, no extras, no missing — any
    failure is a structural fail). Ship this task with ZERO built-in
    patterns registered — T-5 and T-6 register their own patterns on
    top. Unit tests cover encode/decode round-trip, overlapping
    matches, and validator enforcement.

- id: T-5
  title: "Glossary via placeholder mode"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/placeholder.py
      - src/mdpo_llm/__main__.py
      - tests/test_placeholder.py
      - tests/test_batched_processing.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: normal
  depends_on: [T-4]
  brief: .ccx/tasks/T-5.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-5
  branch: ccx/T-5
  started_at: "2026-04-18T08:10:13Z"
  finished_at: "2026-04-18T08:25:50Z"
  exit_status: approved
  notes: |
    Roadmap #5. Add `--glossary-mode=placeholder|instruction` (default
    `instruction` for back-compat in v0.4, flip to `placeholder` in
    v0.5 after the acceptance fixture passes). Placeholder path: match
    glossary terms with word-boundary regex, substitute placeholders
    pre-call, restore with target-language form post-call. Handle
    trailing morphology ("APIs" matching "API") conservatively —
    prefer false-negative (no match) over false-positive (mid-word
    mangle); document the rule in docstrings. Keep the instruction
    path intact.

- id: T-6
  title: "Anchor + inline HTML placeholder protection"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/placeholder.py
      - src/mdpo_llm/validator.py
      - tests/test_placeholder.py
      - tests/test_batched_processing.py
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: normal
  depends_on: [T-4]
  brief: .ccx/tasks/T-6.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-6
  branch: ccx/T-6
  started_at: "2026-04-18T08:26:57Z"
  finished_at: "2026-04-18T10:22:10Z"
  exit_status: approved
  exit_notes: "human-approved via Discord despite cap-hit at review 20/20 with 2 P2 findings fixed post-final-review"
  notes: |
    Roadmap #6. Register placeholder patterns for `{#anchor-id}` and
    inline raw HTML attribute strings (`class="bare"` etc.) — protecting
    link-attribute strings that have been observed to mangle in the
    real-world test. Placeholders are always active (no opt-out flag);
    validator failure on a missing anchor is a structural fail, not a
    warning. Regression fixture: a source block with 5 anchors + 3
    class-attr HTML links round-trips every placeholder unchanged.

- id: T-7
  title: "Refine mode + inplace deprecation"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/prompts.py
      - src/mdpo_llm/validator.py
      - src/mdpo_llm/__main__.py
      - src/mdpo_llm/__init__.py
      - tests/test_batched_processing.py
      - tests/test_validator.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: normal
  depends_on: [T-4]
  brief: .ccx/tasks/T-7.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-7
  branch: ccx/T-7
  started_at: "2026-04-18T10:25:24Z"
  finished_at: "2026-04-18T11:49:30Z"
  exit_status: approved
  notes: |
    Roadmap #9. Split into two modes: `translate` (cross-language,
    today's behaviour) and `refine` (same-language clarity/grammar
    polish, no language change). Shared core (parsing, PO, batch,
    reference pool, placeholders from T-4); mode selects prompt and
    validator config. Refine writes to a separate `refined_path`; the
    original source and its PO `msgid` are never overwritten. Add
    `translate --refine-first` composition. Validator: refine drops the
    target-language-presence check and adds a language-stability check
    (output dominant language must match source). Emit DeprecationWarning
    on `inplace=True` pointing users to refine mode; schedule removal
    in v0.5.

- id: T-8
  title: "Batch concurrency within a single file (experimental)"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/batch.py
      - src/mdpo_llm/__main__.py
      - tests/test_batch.py
      - tests/test_batched_processing.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: low
  depends_on: [T-1]
  brief: .ccx/tasks/T-8.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-8
  branch: ccx/T-8
  started_at: "2026-04-18T11:50:24Z"
  finished_at: "2026-04-18T12:16:40Z"
  exit_status: approved
  notes: |
    Roadmap #3. Expose `--batch-concurrency N` that allows N batches
    within a single file to fly in parallel after the first 1–2
    batches have seeded the reference pool. Off by default. T-1 must
    land first so we have before/after wall-clock + token numbers to
    justify the complexity — do NOT start this before the baseline is
    comparable.

- id: T-9
  title: "Multi-target translation in a single call"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/prompts.py
      - src/mdpo_llm/batch.py
      - src/mdpo_llm/__main__.py
      - src/mdpo_llm/__init__.py
      - tests/test_batch.py
      - tests/test_batched_processing.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: low
  depends_on: [T-5, T-6, T-7]
  brief: .ccx/tasks/T-9.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-9
  branch: ccx/T-9
  started_at: "2026-04-18T12:16:57Z"
  finished_at: "2026-04-18T13:02:30Z"
  exit_status: approved
  notes: |
    Roadmap #7. `target_langs: list[str]`. One batched call per source
    block returns {block_id: {lang: translation}}. Shared source-side
    decomposition (same placeholder replacements, same glossary hits)
    across languages — only the LLM call fans out. Per-lang reference
    pools, per-lang PO files. Before building this, first benchmark
    the cheaper "canonical-seeded" alternative: run translate for one
    "anchor" language, seed other languages' reference pools from the
    anchor's output. If the canonical-seeded run closes the consistency
    gap, close this task as superseded.

- id: T-10
  title: "Filename translation (optional mode)"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/parser.py
      - src/mdpo_llm/__main__.py
      - tests/test_batched_processing.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: low
  depends_on: []
  brief: .ccx/tasks/T-10.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-10
  branch: ccx/T-10
  started_at: "2026-04-18T13:02:40Z"
  finished_at: "2026-04-18T13:48:50Z"
  exit_status: approved
  notes: |
    Roadmap #6b. Opt-in `--translate-paths` flag. Filenames become
    their own "block type" tracked in a separate `_paths.po` per
    target_dir. Emit `path_map.json` alongside the translated tree so
    downstream tooling (link rewriters, sitemaps, CI) can resolve
    source → target paths. Existing link text/URLs in content must NOT
    be auto-rewritten by this task — that's a separate cross-reference
    problem and scoping it here would invalidate every translated
    doc's internal links.

- id: T-11
  title: "Per-directory glossary with parent-chain resolution"
  scope:
    include:
      - src/mdpo_llm/processor.py
      - src/mdpo_llm/__main__.py
      - tests/test_glossary_cascade.py
      - README.md
      - CHANGELOG.md
    exclude: []
  status: merged
  priority: normal
  depends_on: []
  brief: .ccx/tasks/T-11.md
  attempts: 1
  worktree: /home/will/Repositories/mdpo-llm-T-11
  branch: ccx/T-11
  started_at: "2026-04-19T02:53:50Z"
  finished_at: "2026-04-19T03:26:30Z"
  exit_status: approved
  notes: |
    Multi-document translation often wants the SAME term handled
    differently per subtree: `docs/api/` preserves "API" verbatim,
    `docs/marketing/` lets the LLM translate it naturally. Today the
    processor pins a single `self._glossary` at __init__ so this is
    impossible without re-instantiating per file.

    Add per-directory glossary discovery: when `process_directory`
    encounters a file, walk parents from the file's directory up to
    the tree root collecting every `glossary.json` it sees, plus
    cwd's `./glossary.json`, plus the CLI-supplied `--glossary`
    override. Merge parent → child so CHILD WINS per term, and
    support a `"__remove__"` sentinel value that unsets a term
    inherited from a parent (the child then lets the LLM translate
    it freely). A term absent from a level inherits from its parent;
    `null` / string values follow existing semantics (do-not-
    translate / force specific translation).

    Merge rule (parent → child iteration):
        for level in chain:
            for term, value in level.items():
                if value == "__remove__":
                    result.pop(term, None)
                else:
                    result[term] = value

    Per-file resolution MUST cache by directory so the same parent
    chain isn't re-read once per file in a deep tree. Placeholder
    mode (now the default) plays nicely with per-file glossaries —
    the glossary does not appear in the system prompt, so prompt
    cache hit rate stays stable even when every file has a slightly
    different effective mapping.

    CLI: no new flag. Auto-detection runs whenever `--glossary` is
    NOT passed; explicit `--glossary PATH` keeps its current
    single-file-override semantics (used as the TOPMOST / CLOSEST
    level, overriding any discovered chain). Emit one INFO log per
    file under `-v` listing the resolved chain so users can debug
    cascade behaviour.

    Tests (new `tests/test_glossary_cascade.py`):
      - walk finds nearest parent glossary.json, merges with ancestors
      - `__remove__` sentinel unsets inherited term
      - child overrides parent value for same term
      - term absent from child inherits from parent
      - `--glossary` CLI override takes precedence over discovered chain
      - directory cache avoids re-reading parent files for siblings
      - empty chain → glossary disabled (existing behavior preserved)

    README: extend `## Glossary` with a `### Per-directory glossary
    cascade` subsection showing the layout + merge rule + sentinel.
    CHANGELOG: `### Added` entry under `## Unreleased`.
```
