# mdpo-llm

[![Python Version](https://img.shields.io/pypi/pyversions/mdpo-llm.svg)](https://pypi.org/project/mdpo-llm/)
[![PyPI Version](https://img.shields.io/pypi/v/mdpo-llm.svg)](https://pypi.org/project/mdpo-llm/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![License](https://img.shields.io/pypi/l/mdpo-llm.svg)](https://github.com/willysk73/mdpo-llm/blob/main/LICENSE)

**Translate Markdown with LLMs — and only pay for what changed.**

mdpo-llm splits your Markdown into blocks, tracks each one in a PO file, and sends only new or changed blocks to your LLM. Edit one paragraph in a 50-block document? One API call, not fifty.

## What's new in v0.5

- **LLM validation + bounded retry loop** (T-16). Opt in with
  `validation="llm"` to add a second-pass grader LLM that scores each
  translated batch against the source. Failed keys retry with the
  full history of rejection reasons appended to the system prompt
  (`**PREVIOUS ATTEMPT REJECTED — REASONS:**`); at retry index
  `ceil(max_retries / 2)` the loop swaps to `fallback_model` if one
  is configured. Residual failures after the budget exhausts are
  marked fuzzy with the last reason in tcomment. Defaults: 3 retries,
  no fallback model. Configure via `--max-retries` /
  `--fallback-model` (or the matching constructor kwargs).
  Structural `conservative` checks still run as a cheap pre-gate.
- **Free-text domain context injection** (T-18). `--context PATH`
  reads a UTF-8 text file and injects it into every translation /
  validation system prompt under a stable `**ADDITIONAL CONTEXT...**`
  header. Per-directory `context.md` files cascade parent → child
  before the override is appended, so document-level briefing
  (audience, tone, conventions, proper nouns) flows alongside the
  glossary's term-level substitutions. Empty / missing files at any
  level are silently skipped. See *Domain context* below.

## What's new in v0.3

- **Batched JSON-mode translation** (default on). A 50-block first-run collapses from 50 serial calls to ~2 batched calls.
- **Intra-document consistency by construction** — sibling blocks share one LLM context, so tone and terminology unify across a document.
- **Post-translation validator** (opt-in via `validation="conservative"` or `"strict"`) checks heading levels, fence counts, glossary preservation, and target-language presence.
- **Prompt caching** hint on the stable system prefix — pass `enable_prompt_cache=True` to cut cost on re-runs and large directory jobs.
- **CLI** — `python -m mdpo_llm translate|translate-dir|estimate|report …`.
- **Typed result dataclasses** (`ProcessResult`, `Coverage`, `BatchStats`) with dict-style access for backward compatibility.
- **Dry-run estimator** — `processor.estimate(src)` reports pending blocks and estimated tokens with zero API calls.
- **Per-run receipt** — every `translate` / `translate-dir` run attaches a `Receipt` with total tokens, per-1M USD pricing, wall-clock duration, and API-call count. CLI prints a human-readable block to stderr; `--json-receipt PATH` dumps the same data as JSON for CI.
- **Progress display** — `translate` / `translate-dir` render a live `rich` progress bar on a TTY (batches for a single file, file count for a directory). Auto-suppressed under `-v`, when stderr isn't a TTY, when `MDPO_NO_PROGRESS` is set, or via `--no-progress`. The library stays UI-agnostic: pass `progress_callback=` to `MdpoLLM(...)` to receive `ProgressEvent` dataclasses and render your own UI.

v0.2 behaviour (one call per block) is preserved via `batch_size=0`.

## How It Works

```mermaid
flowchart LR
    A["Markdown\nSource"] --> B["Parse\ninto blocks"]
    B --> C["Track\nin PO file"]
    C --> D{"Changed?"}
    D -- Yes --> E["Send to\nLLM"]
    D -- No --> F["Reuse existing\ntranslation"]
    E --> G["Reconstruct\nMarkdown"]
    F --> G
```

Each block (heading, paragraph, code block, list, table) is tracked independently. On subsequent runs, only blocks whose source text changed get sent to the LLM — the rest are served from the PO cache.

### Incremental processing in practice

```
First run:    8 blocks parsed → 8 API calls → full document translated
Edit source:  change 1 paragraph
Second run:   8 blocks parsed → 1 API call  → only the changed block retranslated
```

## Translation Context

Blocks aren't translated in isolation. As each block is translated, it's added to a reference pool. Subsequent blocks receive the most similar previous translations as few-shot examples, so the LLM maintains consistent tone, terminology, and style across the entire document.

```
Block 1: "Introduction"     → translated (no context yet)
Block 2: "Getting Started"  → translated with Block 1 as reference
Block 3: "Installation"     → translated with Blocks 1–2 as reference
...
```

On re-runs, the pool is seeded from all existing translations in the PO file, so even a single changed paragraph benefits from the full document's context.

## Installation

```bash
pip install mdpo-llm
```

## Quick Start

### 1. Translate a document

No subclassing, no boilerplate. Pass a model string and go.

```python
from pathlib import Path
from mdpo_llm import MdpoLLM

processor = MdpoLLM(
    model="gpt-4",            # any LiteLLM model string
    target_lang="ko",         # baked into the system prompt
    temperature=0.3,          # forwarded to litellm.completion()
)

result = processor.process_document(
    source_path=Path("docs/README.md"),
    target_path=Path("docs/README_ko.md"),
    # po_path defaults to docs/README_ko.po
)

print(f"Processed {result['translation_stats']['processed']} blocks")
print(f"Coverage: {result['coverage']['coverage_percentage']}%")
```

Run it again after editing the source — only the changed paragraphs get reprocessed.

### 2. Process a directory

```python
result = processor.process_directory(
    source_dir=Path("docs/"),
    target_dir=Path("docs_ko/"),
    glob="**/*.md",
    max_workers=4,  # files processed concurrently
    # po_dir defaults to target_dir (PO files next to translated files)
)

print(f"{result['files_processed']} files processed")
print(f"{result['files_skipped']} files unchanged")
```

The directory structure is mirrored into `target_dir`. Each file gets its own PO file and its own reference pool. By default, PO files are placed next to the target files; pass `po_dir` to store them separately.

#### Optional: translate filenames too (`--translate-paths`)

By default `process_directory` mirrors the source tree 1:1, so a document at `docs/guide/intro.md` ends up at `docs_ko/guide/intro.md`. Opting into `--translate-paths` (CLI) or `translate_paths=True` (API) additionally translates the filesystem path segments themselves — directory names and markdown file stems — so the target tree uses localized filenames:

```bash
python -m mdpo_llm translate-dir docs/ docs_ko/ \
    --model gpt-4o \
    --target ko \
    --po-dir po/ \
    --translate-paths
```

What this produces:

- **`_paths.po`** — a dedicated catalog under `--po-dir` (or `target_dir` when `--po-dir` is omitted) that stores one entry per distinct source segment. Segment translations flow through the same LLM pipeline as content blocks, so caching, glossary configuration, and token receipts behave the same way. Re-running the command hits cache on unchanged segments and spends zero API calls on them.
- **`path_map.json`** — a JSON map `{ "source/relative.md": "translated/relative.md", ... }` written at the root of the translated tree. Downstream tooling (link rewriters, sitemap generators, CI jobs) can read this file to resolve the source ↔ target pairing without re-running the translator.
- **Sanitized, deterministic slugs** — LLM output is NFC-normalised, whitespace is collapsed, and characters reserved on Windows / POSIX filesystems (`/\\<>:"|?*` plus control bytes) are stripped. If two sibling source files end up with the same translated slug, `-2` / `-3` disambiguators are appended in alphabetical source order so the output is reproducible. File extensions are preserved verbatim. Dotfile segments (`.github`, `.well-known`) pass through unchanged so CI and web-infrastructure paths don't silently break.
- **PO files stay keyed on the SOURCE path.** Per-file `.po` outputs under `--po-dir` are still laid out using the source-relative path, so incremental re-runs hit the same PO cache even when the target filename moves between runs.

What it explicitly does NOT do:

- **Link rewriting is out of scope.** Markdown link text and URLs inside translated content are not modified — auto-rewriting them would invalidate every document's internal anchors and cross-references. `path_map.json` is published so downstream tooling can do that rewrite deterministically in a subsequent pass.

### 3. Use any provider

LiteLLM supports 100+ providers. Just change the model string:

```python
# OpenAI
MdpoLLM(model="gpt-4", target_lang="ko")

# Anthropic
MdpoLLM(model="anthropic/claude-sonnet-4-5-20250929", target_lang="ko")

# Google
MdpoLLM(model="gemini/gemini-pro", target_lang="ko")

# Azure OpenAI
MdpoLLM(model="azure/my-deployment", target_lang="ko", api_base="https://...")
```

## Language Handling

### `target_lang` — tell the LLM which language to produce

A BCP 47 locale string (e.g. `"ko"`, `"ja"`, `"zh-CN"`) baked into the system prompt. The source language is auto-detected by the LLM — you only specify the target.

```python
processor = MdpoLLM(model="gpt-4", target_lang="ja")
```

When `target_lang` is set, new PO files will include a `Language` header (e.g. `Language: ja`).

## Glossary

Protect brand names, trademarks, and proper nouns from translation — or force specific translations for them.

### Inline glossary

```python
processor = MdpoLLM(
    model="gpt-4",
    target_lang="ko",
    glossary={
        "GitHub": None,                # None = do not translate
        "Markdown": None,
        "pull request": "풀 리퀘스트",  # force specific translation
        "API": "API",
    },
)
```

### JSON glossary file

For multi-locale projects, keep a single `glossary.json`:

```json
{
  "GitHub": null,
  "Markdown": null,
  "pull request": {
    "ko": "풀 리퀘스트",
    "ja": "プルリクエスト"
  },
  "API": "API"
}
```

- `null` — do not translate (any locale)
- `"string"` — use this translation for all locales
- `{"ko": "...", "ja": "..."}` — per-locale; if the current locale isn't listed, the term is kept as-is

```python
processor = MdpoLLM(
    model="gpt-4",
    target_lang="ko",
    glossary_path="glossary.json",
)
```

If both `glossary` and `glossary_path` are provided, inline entries override the file.

Only glossary terms that actually appear in each block are injected into the prompt, so a large glossary doesn't waste tokens on irrelevant blocks.

See [`examples/glossary.json`](examples/glossary.json) for a full example with brand names, technical terms, and per-locale translations.

### Glossary mode: `instruction` vs `placeholder`

`glossary_mode` (constructor kwarg, CLI `--glossary-mode`) controls how
glossary terms reach the model:

- `"placeholder"` (default): substitutes every glossary term with an
  opaque `⟦P:N⟧` token **before** the call and restores the target-
  language form (or the original term for do-not-translate entries)
  **after** the call. The model never sees the terms, so it cannot
  translate, renumber, or mangle them — and the round-trip check
  automatically flags any dropped token.
- `"instruction"`: appends a glossary block to the system prompt. The
  LLM sees the raw source text and is asked to preserve or translate
  each term as specified. Use this when your terms contain characters
  that `"placeholder"` cannot match (see caveats below).

```python
processor = MdpoLLM(
    model="gpt-4",
    target_lang="ko",
    glossary={"GitHub": None, "pull request": "풀 리퀘스트"},
    glossary_mode="placeholder",
)
```

Matching is **case-sensitive word-boundary** (`\bterm\b`). Trailing
morphology is NOT matched: `"APIs"` does not match a glossary term
`"API"` because the trailing `s` breaks the word boundary. This is a
deliberate false-negative — a mid-word false-positive would corrupt
neighbouring text, while a missed match simply falls through to the
LLM's normal translation path. Terms whose first or last character
isn't a word character (e.g. `.NET`, `C++`) are silently skipped for
the same reason; use `"instruction"` mode when those matter.

### Per-directory glossary cascade

`translate-dir` (and any `process_directory` call) auto-discovers a
glossary chain per file so different subtrees can treat the same term
differently without re-instantiating the processor. For every source
file, the resolver walks from the tree root down to the file's
directory, layering each `glossary.json` it finds, then applies
`./glossary.json` from the current working directory, then the
`--glossary PATH` override (topmost).

Layout:

```text
docs/
├── glossary.json          # root: baseline terminology
├── api/
│   ├── glossary.json      # api/: preserve "API" verbatim
│   └── reference.md
└── marketing/
    ├── glossary.json      # marketing/: let the LLM translate "API"
    └── landing.md
```

Merge rule (parent → child, **CHILD WINS** per term):

```json
// docs/glossary.json
{ "API": "API", "GitHub": null }

// docs/marketing/glossary.json — unset the inherited "API" mapping
{ "API": "__remove__" }
```

After merge for `docs/marketing/landing.md`: `{"GitHub": null}` — the
root's `GitHub` do-not-translate entry is inherited, but `API` has
been removed so the LLM is free to translate it contextually.

A term missing from a child level **inherits** from its parent; a
`null` or string value follows the existing semantics
(do-not-translate / force a specific translation); a `"__remove__"`
sentinel value unsets the inherited term. Per-locale dicts
(`{"ko": "풀 리퀘스트", "ja": "プルリクエスト"}`) are resolved at each
level before merging.

The CLI needs no new flag. Auto-detection runs whenever `--glossary`
is not passed; supplying `--glossary PATH` keeps its single-file
override semantics but sits on top of the discovered chain, so one
flag can still force a term for every file regardless of the
subtree. Under `-v` a one-line INFO log per file names every
`glossary.json` in the resolved chain so a surprising substitution
can be debugged without rerunning.

Directory-level caching resolves each ancestor's `glossary.json`
exactly once — sibling files in the same subtree reuse the cached
merged chain.

## Domain context (`--context`)

Glossary handles term-level substitutions; **`--context`** handles
the document-level brief — audience, tone, conventions, proper-noun
guidance — that does not fit cleanly as term pairs. The flag accepts a
path to a UTF-8 text file (any format; treated as opaque text) whose
contents are appended verbatim to the system prompt under a stable
header:

```text
**ADDITIONAL CONTEXT (use for proper nouns, terminology, tone, audience):**
{file content verbatim}
```

```bash
python -m mdpo_llm translate-dir docs/ out/ \
  --model gpt-4o --target ko \
  --context briefs/game-security.md
```

**Cascade rules** (mirrors the glossary cascade in shape, but
concatenates instead of overriding):

- Per-directory `context.md` files in the source tree are walked
  parent → child and **concatenated** (child *appends* to parent — a
  child writer typically wants to extend the parent's framing, not
  replace it).
- A `context.md` in the current working directory (when not already
  in the tree walk) is appended after the tree cascade.
- `--context PATH` is appended **last** as the topmost / closest
  layer.
- Empty / missing files at any level are silently skipped — most
  directories will not have a `context.md`, and warning on absence
  would be noise.
- `context.md` files are **excluded from the source glob** in
  `translate-dir` / `refine-dir` whenever the glob *also* matches
  non-context files — the cascade configuration is treated as
  configuration, not as translatable content. A glob that targets
  only `context.md` (for example `**/context.md`) is respected
  verbatim so deliberate callers can still translate them as
  documents.

Both `translate` / `translate-dir` and `refine` / `refine-dir`
honour `--context`; the same brief flows into the LLM-validator
prompt under `validation=llm` so the validator grades against the
same domain framing the translator saw.

> **Token-cost note.** The resolved context is appended to every
> system prompt of every batch. A large file (KB+) inflates token
> usage proportionally on every API call — keep the brief tight, or
> rely on prompt caching (`--prompt-cache`) so the stable prefix is
> reused across batches.

## Auto source-language bracket placeholders

Bracket tokens that hold source-language identifiers — `<전송>`,
`{게임코드}`, `<1단계>`, `/users/{한글id}` path parameters — are
auto-registered on the per-file placeholder registry before every
LLM call, so they survive a translate pass verbatim without the
caller having to enumerate each one in `glossary`. On by default;
pass `auto_bracket_placeholders=False` to turn it off.

Detection rule — a Unicode word character **outside the target
language's primary script** inside a single-angle `<…>` or
single-brace `{…}` span whose content is identifier-shaped (word
characters plus `-`, `_`, `.`; no whitespace, no punctuation). The
target-script lookup is keyed on the BCP 47 primary-language prefix
of `target_lang` (e.g. `ko`/`ja`/`zh` → CJK, `ru`/`uk`/`bg` →
Cyrillic, `ar`/`fa` → Arabic, `en`/`fr`/`de`/unknown → Latin/ASCII),
so a Korean refine pass (`target_lang="ko"`) correctly leaves
`{전송}` and `<다음>` for the refiner, while an English target
protects the same spans as before. Mixed-script identifiers like
`<id_게임코드>` still match for a Korean target because the Latin
prefix is non-target-script content. A bracket whose content is
entirely target-script (`{page_id}` with English target, `<전송>`
with Korean target) and multi-word UI labels (`{상태 변경}`,
`<확인 버튼>`) flow through the translate prompt normally — callers
who need to pin those specifically can register their own pattern
via `placeholders=PlaceholderRegistry(...)`.

The regex is deliberately conservative so it does not over-protect:

- **`{{…}}` Mustache / Jinja templates** are excluded — a lookbehind
  / lookahead guard keeps the inner single-brace from being
  tokenized away from under the template engine.
- **Real HTML opening / closing tags** (`<a href="/한글">`,
  `</한글>`, `<!-- 한글 -->`, `<?xml ?>`) are excluded — the
  `html_attr` built-in retains its allowlist-based protection
  contract so translatable attributes like `title` / `alt` /
  `aria-label` still reach the translate prompt.
- **Inline code** (`` `{한글}` ``) is skipped so documentation that
  illustrates bracket syntax literally does not freeze the example.
- **Caller-supplied glossary entries win** — when a glossary term
  sits inside the bracket span, the glossary pattern tokenizes the
  inner term and auto-register defers; the LLM sees the bracket
  structure with the glossary token inside. Useful when a term has
  an explicit target-language mapping (`{"게임코드": "GameCode"}`)
  and the bracket structure should still be translatable around it.

```python
processor = MdpoLLM(
    model="gpt-4",
    target_lang="en",
    auto_bracket_placeholders=True,  # default
)
# Source "/api/{게임코드}/profile" round-trips byte-for-byte;
# the LLM never sees "게임코드" and cannot rewrite it.
```

Opt out per instance via `auto_bracket_placeholders=False`, or
globally for ops overrides via the `MDPO_AUTO_BRACKET_PLACEHOLDERS`
env var (`1` / `0` / `true` / `false` / `yes` / `no` / `on` / `off`,
case-insensitive). Unrecognised env values fall through to the
kwarg so typos don't silently flip behaviour.

Dedicated custom placeholder patterns (via
`placeholders=PlaceholderRegistry(...)`) take priority over
auto-register on exact-span ties, so a caller who has their own
protection for a specific shape can still override the default
pattern without disabling the feature globally.

### Custom placeholder rules (advanced)

For shapes auto-bracket and glossary do not catch — environment
variable references (`${VAR}`), backtick + script-class tokens, custom
DSL brackets, etc. — the CLI accepts `--placeholder-rules rules.json`
on every LLM-issuing subcommand (`translate`, `translate-dir`,
`translate-multi`, `refine`, `refine-dir`). The file is a flat JSON
array of rule objects:

```json
[
  {"name": "env_var_refs", "regex": "\\$\\{[A-Z_]+\\}"}
]
```

Each rule needs a non-empty string `name` and a Python `re` `regex`
string; any extra field is rejected so a typo like `pattern` instead
of `regex` fails the run instead of silently producing a no-op.
Regexes are compiled eagerly so a malformed pattern surfaces with
exit code 2 before any LLM call. Rules compose with the existing
registration order (caller `placeholders` ⟶ T-6 anchors / `html_attr`
⟶ glossary ⟶ T-14 auto-bracket); a glossary entry covering the same
span still wins on decode.

Reach for this only when neither auto-bracket (T-14) nor glossary
covers your token shape — those should be your first stop.

### Residue post-processing (advanced)

`--residue-pass on` (T-17) adds an opt-in *post-translation* sweep
that detects source-language characters left inside fenced code
blocks or inline code spans and re-translates only the affected
spans through specialised prompts:

- Fenced code block → preserve identifiers + comments, translate
  user-facing string literals only.
- Filename-shaped inline code (e.g. `회원목록.md`) →
  transliterate to `UPPER_SNAKE_CASE` ASCII.
- Other inline code → translate the source-language text to the
  target locale.

```bash
python -m mdpo_llm translate \
  --model gpt-4o \
  --target en \
  --residue-pass on \
  source.md target.md
```

The pass runs AFTER LLM validation so it sees the final committed
`msgstr`, skips entries already marked fuzzy by the retry budget
(re-running known-bad output is waste), and skips refine mode
entirely (refine is same-language, so "source-language residue" is
undefined). It is best-effort: any failure (LLM exception,
post-repair placeholder-token round-trip rejection) keeps the
pass-1 translation verbatim and logs a warning.

Default is `off` pending soak time. False-positive risk on
edge-case docs (mixed-script identifiers, source script kept
intentionally for branding) is low but real, so the flag is opt-in
until a release of real-world use settles its sensitivity.

## LLM validation + bounded retry loop (T-16)

The default `validation="conservative"` / `"strict"` checks are
cheap structural assertions (heading levels match, fence counts
match, glossary preservation holds). They catch shape regressions
but not subtle quality issues like a translation that picked the
wrong term, dropped a clause, or quietly left a sentence in the
source language.

`validation="llm"` (opt-in) adds a *second* LLM pass that grades
each translated batch against its source and retries the failed
keys only:

```bash
python -m mdpo_llm translate \
  --model gpt-4o \
  --target ko \
  --validation llm \
  --max-retries 3 \
  --fallback-model "anthropic/claude-sonnet-4-5-20250929" \
  source.md target.md
```

Pipeline per batch:

1. Translate (existing path).
2. Run the structural validator as a cheap pre-gate.
3. Send each `{source, output}` pair to a validator LLM that returns
   `{key: {binary_score, reason}}` via JSON mode.
4. Partition pass / fail keys.
5. Retry only the failed keys; the **full history** of rejection
   reasons is appended to the system prompt under
   `**PREVIOUS ATTEMPT REJECTED — REASONS:**`.
6. At retry index `ceil(max_retries / 2)`, the loop swaps to
   `--fallback-model` (when one is configured).
7. Re-grade the retry candidates.
8. After `max_retries` retries, residual failures are marked fuzzy
   with the **last** rejection reason recorded in `tcomment`.

Tunables:

- `--max-retries N` (default `3`, clamped to `0..10`). `N=0` runs
  the grader once and marks any failure fuzzy without retrying;
  larger values trade tokens for quality.
- `--fallback-model MODEL` (default unset). When unset, every
  retry stays on `--model`. When set, the swap fires at the
  midpoint of the retry budget so the second half of attempts
  uses the alternate model — useful when the primary model
  consistently misses a class of translations and a different
  model is more likely to recover.
- `validation="llm"` implies the structural `conservative` checks;
  you don't need to run a separate `validation="conservative"` pass.

Reference pool on retry:  every key that has *already passed* in
this batch becomes a few-shot example for the keys that have not
— a free intra-batch consistency signal at zero extra LLM cost.

Multi-target (`process_document_multi`):  each language fans out
to its own validator call (the grader judges in that language's
context) and runs an independent retry budget per lang.

Cost note:  LLM validation roughly doubles input tokens per batch
and adds output tokens. Expected use is publishing / CI flows;
`validation="conservative"` (or `"off"`) stays the right default
for daily iterative work.

## Refine mode

`mode="refine"` polishes a Markdown document in its **original** language:
fixes grammar, tightens phrasing, smooths flow — without translating or
switching languages. It reuses the translate pipeline — parsing, PO
tracking, batching, reference pool, placeholders — and swaps in a
refine-specific prompt and validator configuration.

Key contract:
- Refine **never** overwrites the source or its PO `msgid`. The refined
  output goes to a separate `refined_path` (or the `target_path` you
  supply); `msgstr` holds the refined text, `msgid` keeps the original.
- `target_lang` names the source/output language (refine is
  same-language by definition).
- The validator drops the target-language-presence check and adds a
  `language_stability` check: if the source detects as one language and
  the refined output as another, the entry is flagged fuzzy.
- `inplace=True` is incompatible with refine and raises.

```python
from mdpo_llm import MdpoLLM

refiner = MdpoLLM(
    model="gpt-4",
    target_lang="en",    # refine preserves the source language
    mode="refine",
)
refiner.process_document(
    source_path="docs/README.md",
    target_path="docs/README.refined.md",   # refined output
    po_path="docs/README.refined.po",
)
```

From the CLI:

```bash
mdpo-llm refine docs/README.md docs/README.refined.md --model gpt-4 --target en
mdpo-llm refine-dir docs/ docs_refined/ --model gpt-4 --target en
```

### `translate --refine-first` composition

When the upstream source is noisy (typos, bad grammar, inconsistent
phrasing), polish it first, then translate. Both passes contribute
tokens to the receipt; the refined intermediate lives at
`--refined-path` so downstream re-runs can reuse it.

`refine_lang` / `--refine-lang` is **required** — it names the BCP 47
locale of the source document, which is what the refine pass must
preserve. There is no safe default: using `target_lang` would pin the
refine pass to the translation TARGET and the cross-language run would
collapse into same-language nonsense.

Refine-first requires **distinct** paths and POs for the two passes —
`refined_path` ≠ `target_path`, `refined_po_path` ≠ `po_path`.
Sharing either would let the translate pass see the refine output as
"already processed" and skip translation entirely. On the first
refine-first run with a pre-existing translate PO, the translate PO is
re-keyed on refined msgids (the source changed, so prior source-keyed
entries are obsoleted by design); the translate pass still seeds its
reference pool with the old `(msgid, msgstr)` pairs so tone and
terminology survive as few-shot context.

```bash
mdpo-llm translate docs/README.md docs/README_ko.md \
    --model gpt-4 --target ko \
    --refine-first --refined-path docs/README.refined.md --refine-lang en
```

```python
processor = MdpoLLM(model="gpt-4", target_lang="ko")
processor.process_document(
    source_path="docs/README.md",
    target_path="docs/README_ko.md",
    refined_path="docs/README.refined.md",
    refine_first=True,
    refine_lang="en",
)
```

## Multi-target translation in a single call (experimental)

`process_document_multi` translates one Markdown source into several
languages in a single batched LLM call per source group. Source-side
decomposition — placeholder substitution, reference lookup, glossary
matching — runs ONCE per block regardless of the number of target
languages, so the input-token bill is amortised across every target
while only output tokens grow with `len(target_langs)`.

```python
from pathlib import Path
from mdpo_llm import MdpoLLM

processor = MdpoLLM(
    model="gpt-4o",
    target_lang="ko",   # ignored by process_document_multi; constructor-required
    batch_size=40,
)

result = processor.process_document_multi(
    source_path=Path("docs/README.md"),
    target_langs=["ko", "ja", "zh-CN"],
    target_paths={
        "ko": Path("docs/ko/README.md"),
        "ja": Path("docs/ja/README.md"),
        "zh-CN": Path("docs/zh-CN/README.md"),
    },
    # po_paths defaults to each target with a .po suffix
)
print(result["receipt"].render())
for lang, pr in result["by_lang"].items():
    print(lang, pr["translation_stats"]["processed"])
```

From the CLI:

```bash
mdpo-llm translate-multi docs/README.md \
    --target-template "docs/{lang}/README.md" \
    --langs ko,ja,zh-CN \
    --model gpt-4o
```

Contract:

- Each target language has its OWN PO file and OWN reference pool —
  translations do not cross languages. The pool is seeded per-lang
  from the respective PO on load.
- Per-language distinctness is enforced: `target_paths` / `po_paths`
  must resolve to distinct paths per lang, and neither may alias the
  source path. Colliding paths fail up front with a `ValueError` so
  automation gets a clean usage error rather than a mid-run clobber.
- `mode="refine"` is rejected — refine is same-language by contract
  and multi-target only makes sense for translate.
- `inplace=True` is NOT supported: overwriting one source msgid with
  N different-language translations is undefined.
- Partial per-lang coverage in the model's response is tolerated.
  Any languages that came back with well-typed strings commit
  directly; missing langs per block fall back to a single-target
  per-entry call so the PO is never left half-populated.
- A single `Receipt` is returned at the top level; each per-lang
  `ProcessResult` has `receipt=None` because tokens are billed ONCE
  across the whole run. `receipt.target_lang` is a comma-joined list
  for operator auditability.

### Canonical-seeded alternative

Before adopting multi-target for consistency, consider the cheaper
"canonical-seeded" baseline: run single-target `translate` for one
"anchor" language first, then run `translate` for each other language
independently. Consistency comes from each run's own reference pool
seeded from its PO file (which accumulates across re-runs), not from
cross-language sharing. Compare the two approaches' `Receipt` totals
and PO contents on a representative document to decide which is worth
shipping on your workload — the machinery for both ships in the same
release, and no live benchmarks are required for correctness.

### Batch concurrency (experimental)

`batch_concurrency=N` / `--batch-concurrency N` lets multiple batches
from the same file fly in parallel once the first batch has seeded the
reference pool. Off by default — keep it at `1` for deterministic
v0.4 behaviour. The first section-aware group always runs sequentially
so subsequent workers inherit a warm pool; the remaining groups are
submitted to a thread pool of size `N`.

```bash
mdpo-llm translate docs/README.md docs/README_ko.md \
    --model gpt-4 --target ko --batch-concurrency 4
```

```python
processor = MdpoLLM(
    model="gpt-4",
    target_lang="ko",
    batch_concurrency=4,  # up to 4 batches in flight after the seed batch
)
```

Caveats:
- Experimental. Compare against `--batch-concurrency 1` using the
  per-run receipt before adopting a higher value — real speed-up depends
  on your provider's rate limits, latency, and token budget.
- Ignored on the sequential path (`--batch-size 0`) and on any document
  that partitions into a single section-aware group.
- Tokens and stats are merged across workers into the same `Receipt` /
  `BatchStats`; the user-visible output is identical to the sequential
  path.
- Progress callbacks are emitted from worker threads — the contract
  already documented for `process_directory` now applies to batched
  single-file runs when concurrency > 1.

### `inplace=True` is deprecated

Passing `inplace=True` emits a `DeprecationWarning` pointing at refine
mode; the flag is scheduled for removal in v0.5. If you were using
`inplace=True` to "rewrite the source in place after translating",
switch to `mode="refine"` with an explicit `refined_path` — it captures
the intent without clobbering the original document.

## Comparison

| | mdpo-llm | mdpo | md-translator | llm-translator |
|---|:---:|:---:|:---:|:---:|
| LLM-powered | Yes | No | Yes | Yes |
| Incremental (block-level) | Yes | Yes | No | No |
| PO file tracking | Yes | Yes | No | No |
| Translation context (few-shot) | Yes | No | Partial | No |
| LLM-agnostic | Yes | — | Multi-provider | OpenAI only |
| Batch directory processing | Yes | Yes | No | No |

**mdpo** pioneered PO-based Markdown translation but targets manual/MT workflows, not LLMs. **md-translator** and **llm-translator** use LLMs but reprocess entire files on every run. mdpo-llm combines both: PO-tracked incremental processing with LLM-powered translation and cross-block context.

## API Reference

### MdpoLLM

Constructor:

```python
MdpoLLM(
    model,                     # any LiteLLM model string (required)
    target_lang,               # BCP 47 string, baked into system prompt (required)
    max_reference_pairs=5,     # max similar pairs passed as few-shot context
    extra_instructions=None,   # appended to the built-in translation prompt
    post_process=None,         # Callable[[str], str] applied to every LLM response
    glossary=None,             # dict[str, str | None] — inline glossary
    glossary_path=None,        # path to JSON glossary file (multi-locale)
    progress_callback=None,    # Callable[[ProgressEvent], None] — see "Progress hook"
    mode="translate",          # "translate" (cross-language) or "refine" (same-language polish)
    batch_concurrency=1,       # experimental: intra-file parallel batches (see "Batch concurrency")
    auto_bracket_placeholders=True,  # auto-protect <cjk>/{cjk} tokens — see "Auto source-language bracket placeholders"
    **litellm_kwargs,          # temperature, api_key, api_base, etc.
)
```

| Method | Description |
|--------|-------------|
| `process_document(source_path, target_path, po_path=None, inplace=False, *, refined_path=None, refine_first=False, refine_lang=None)` | Process a single Markdown file. `po_path` defaults to `target_path` with `.po` extension. `refined_path`, `refine_first`, `refine_lang` drive refine-mode / `translate --refine-first` composition (see "Refine mode"). `inplace=True` is deprecated — emits a `DeprecationWarning` pointing at refine mode; slated for removal in v0.5. Returns a `ProcessResult` with a `.receipt` summarizing tokens, cost, and duration. |
| `process_directory(source_dir, target_dir, po_dir=None, glob, inplace, max_workers, *, refined_dir=None, refine_first=False, refine_lang=None)` | Process a directory tree concurrently. `po_dir` defaults to `target_dir`. The refine / refine-first kwargs mirror `process_document` across every file. Returns a `DirectoryResult` with a `.receipt` aggregated over every file. |
| `get_translation_stats(source_path, po_path)` | Return coverage and block statistics |
| `export_report(source_path, po_path)` | Generate a detailed text report |

### Receipt

Every `process_document` / `process_directory` call attaches a `Receipt`:

```python
result = processor.process_document(src, tgt)
print(result.receipt.render())            # human-readable block (stderr from the CLI)
print(result.receipt.total_tokens)        # int
print(result.receipt.total_cost_usd)      # float | None (None for unpriced models)
print(result.receipt.duration_seconds)    # float (wall clock)
```

Pricing is resolved from `litellm.model_cost`; models not listed there
leave the cost fields `None` and render as `"—"`. From the CLI, pass
`--json-receipt PATH` on `translate` / `translate-dir` to dump the same
structure as JSON for downstream tooling.

### Progress hook

Pass `progress_callback=` to `MdpoLLM(...)` to observe translation
progress from your own UI. The callable receives a `ProgressEvent`
dataclass with `kind`, `path`, `index`, `total`, and `status` fields.
Event kinds:

- `document_start` / `document_progress` / `document_end` — one
  document's work units (batches in batched mode, entries in sequential
  mode). `total` is set on the start event and repeated on every
  progress tick.
- `directory_start` / `file_start` / `file_end` / `directory_end` —
  fired by `process_directory`. `file_end.status` is `"processed"`,
  `"failed"`, or `"skipped"`.

```python
def on_progress(event):
    if event.kind == "document_progress":
        print(f"{event.path}: {event.index}/{event.total}")

processor = MdpoLLM(model="gpt-4", target_lang="ko", progress_callback=on_progress)
```

The library itself imports nothing from `rich` — install the optional
`rich` extra (`pip install mdpo-llm[progress]`) if you want the built-in
CLI progress bar. The CLI auto-suppresses the bar on non-TTY, under
`-v`, via `--no-progress`, or when `MDPO_NO_PROGRESS` is set, so CI
logs stay clean. Callbacks are invoked from worker threads in
`process_directory`; handle thread-safety if they touch shared state.

### Prompts

The `Prompts` class exposes all built-in prompt templates used by the processor:

```python
from mdpo_llm import Prompts

# See the default translation instruction
print(Prompts.TRANSLATE_INSTRUCTION)
```

## Working with PO Files

PO files (GNU gettext) track the state of each content block:

- **Untranslated** — new content, will be sent to the LLM
- **Translated** — completed, reused on subsequent runs
- **Fuzzy** — source changed since last run, will be retranslated
- **Obsolete** — source block was removed, cleaned up automatically

You can inspect and edit PO files with any standard gettext tool (Poedit, Lokalize, etc.).

When `target_lang` is set, new PO files include a `Language` metadata header so tools can identify the target language.

## Development

```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/
```

## License

MIT
