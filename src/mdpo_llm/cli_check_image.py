"""Vision-LLM image residue check (T-22).

``mdpo-llm check-image <image_path_or_dir> --target <lang>`` walks a
single image or a directory of images and asks a vision-capable LLM
whether each image still contains visible text in the language passed
to ``--target``. The verb is designed to catch screenshots whose UI
text was localized in code but whose image asset still ships the
*source-locale* rendering — so the natural usage is to pass the
SOURCE language of the translation as ``--target`` and treat
``contains_target_lang=true`` records as residue. Example: after an
``en → ko`` translation, run ``check-image docs_ko/ --target en`` to
flag any screenshot that still shows English UI text. This class of
residue is invisible to the text-only validators ([cli_lint], the
T-16 LLM grader, the T-17 residue post-pass) by construction.

CLI shape:

    mdpo-llm check-image <image_path_or_dir> --target <lang>
        [--vision-model NAME]
        [--exit-non-zero-on-findings]

Output: a JSON array of ``{path, contains_target_lang, reason}`` records
to stdout, sorted by path for byte-stable output. ``contains_target_lang``
names the language passed via ``--target``; in the residue workflow
``true`` means "residue detected". ``--exit-non-zero-on-findings`` flips
the exit code to ``1`` when any image is flagged so the verb can gate
CI without callers needing to parse the JSON themselves.

mdpo-llm routes the vision call through ``litellm`` so every other
``mdpo-llm`` verb's model-string contract (OpenRouter, Anthropic,
Bedrock, etc.) keeps working without a separate API client.

Out of scope here:
  * Anything that mutates the scanned tree (PO writes, file rewrites,
    image edits). The verb is read-only by design.
  * Translating the residual text inside the image — that requires OCR
    + an image editor and belongs in a separate pipeline.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import litellm


# Lower-cased image extensions the directory walk recognises. Excludes
# exotic formats (TIFF, BMP, HEIC) that the vision providers consistently
# reject anyway. Sorted alphabetically so the ``--help`` blurb prints in
# a stable order across Python versions.
IMAGE_EXTENSIONS: Tuple[str, ...] = (
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
)


# Default vision model per the task brief. ``openrouter/`` prefix routes
# through OpenRouter so the same OPENROUTER_API_KEY the rest of the
# pipeline uses works here too; override via ``--vision-model`` for
# direct OpenAI / Anthropic / Bedrock calls.
DEFAULT_VISION_MODEL = "openrouter/openai/gpt-4o"


# Strict OCR system prompt. JSON-quoted so the wire payload is parseable
# JSON if the model echoes it back.
SYSTEM_PROMPT = (
    "You are a strict OCR assistant. Look at the provided image and "
    "decide ONLY whether it contains any text in the requested target "
    "language. If you are unsure, answer false. Respond ONLY with "
    'JSON: {"contains": true|false, "reason": "..."}.'
)


# Cap on response tokens. The wire format is a tiny JSON object
# (``{"contains": bool, "reason": "..."}``), so 200 is plenty even with
# a chatty reason field and bills less when the model ignores the
# JSON-only instruction and tries to add commentary.
_MAX_TOKENS = 200


@dataclass(frozen=True)
class ImageCheckRecord:
    """One image's vision-check result.

    ``path`` is the original argument's path (relative or absolute as
    the caller passed it) so the JSON output is reproducible without
    leaking the operator's working directory. ``contains_target_lang``
    is ``True`` when the vision LLM reports text in the language
    passed via ``--target``; for the documented residue workflow
    (``--target`` = source language of the translation) that means
    the image carries un-localised text and is a finding. ``reason``
    is the LLM's free-text justification, capped only by
    ``_MAX_TOKENS`` upstream.
    """

    path: str
    contains_target_lang: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "contains_target_lang": self.contains_target_lang,
            "reason": self.reason,
        }


def _iter_images(root: Path) -> List[Path]:
    """Return image files under ``root`` (single file or directory walk).

    Sorted by string path so the JSON output is byte-stable across
    runs and platforms with different walk ordering. Symlink loops are
    avoided by ``rglob``'s default refusal to descend through cycles.
    """
    if root.is_file():
        return [root]
    images: List[Path] = []
    for path in sorted(root.rglob("*"), key=lambda p: str(p)):
        try:
            if not path.is_file():
                continue
        except OSError:
            # Broken symlinks raise on ``is_file()``; skip rather than
            # abort the whole scan for one unreachable entry.
            continue
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)
    return images


def _encode_image(path: Path) -> str:
    """Encode an image file as a ``data:`` URL for litellm vision input.

    MIME type is resolved via :mod:`mimetypes` from the file extension;
    falls back to ``image/png`` when the extension isn't registered so
    the wire payload still has a content-type the providers accept.
    """
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type or not mime_type.startswith("image/"):
        mime_type = "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def _parse_response(raw: str) -> Tuple[bool, str]:
    """Parse the vision LLM's JSON reply into ``(contains, reason)``.

    Tolerates a ```` ```json ```` fence around the JSON because some
    providers wrap the response even when the system prompt forbids it.
    Raises :class:`ValueError` on unparseable / non-object payloads —
    the CLI surfaces it as a usage error so the operator sees the raw
    response instead of an opaque KeyError later.

    The ``contains`` field accepts either a real JSON boolean or a
    JSON string that matches a recognised true/false alias
    (``"true"`` / ``"false"`` / ``"yes"`` / ``"no"`` / ``"1"`` /
    ``"0"``, case-insensitive). Any other value rejects with
    :class:`ValueError`: a naked ``bool(...)`` would coerce
    ``"false"`` and ``"no"`` to ``True`` (non-empty string), which
    inverts the residue verdict and would silently fail a CI run
    under ``--exit-non-zero-on-findings``.
    """
    text = (raw or "").strip()
    if text.startswith("```"):
        # Strip a ```json ... ``` fence. The first line may be a
        # language tag (``json``) or the opening brace itself.
        text = text[3:]
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
        if text.endswith("```"):
            text = text[:-3].strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"vision LLM did not return valid JSON: {raw!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(
            f"vision LLM JSON must be an object, got: {payload!r}"
        )
    if "contains" not in payload:
        # A missing field is NOT silently defaulted to False: under the
        # CI residue workflow ``--exit-non-zero-on-findings`` would
        # then treat an empty / malformed response as a clean image,
        # silently masking provider outages and prompt-format
        # regressions. Surface it as a usage error so the operator
        # sees the raw response and can retry / pin a different model.
        raise ValueError(
            "vision LLM JSON is missing the required 'contains' "
            f"field; raw response: {raw!r}"
        )
    contains = _coerce_contains(payload["contains"], raw=raw)
    reason = str(payload.get("reason", ""))
    return contains, reason


# Recognised string aliases for the JSON ``contains`` field. Keeps the
# residue verdict honest when a provider emits ``"contains": "false"``
# instead of a real JSON boolean — a naked ``bool()`` would treat the
# non-empty string as ``True`` and invert the meaning. Sets are
# disjoint by construction; any other string value is a hard reject.
_TRUE_STRINGS: frozenset[str] = frozenset({"true", "yes", "1"})
_FALSE_STRINGS: frozenset[str] = frozenset({"false", "no", "0"})


def _coerce_contains(value: object, *, raw: str) -> bool:
    """Return a strict boolean for the LLM's ``contains`` field.

    Accepts:
      * ``bool`` — passed through verbatim.
      * ``str`` — compared case-insensitively against
        :data:`_TRUE_STRINGS` / :data:`_FALSE_STRINGS`. Anything else
        raises.

    Integer / float values are deliberately rejected — coercing
    arbitrary numbers to booleans would mask a malformed schema.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in _TRUE_STRINGS:
            return True
        if normalised in _FALSE_STRINGS:
            return False
    raise ValueError(
        "vision LLM 'contains' must be a JSON boolean (or a "
        "true/false/yes/no/0/1 string alias), got "
        f"{value!r}; raw response: {raw!r}"
    )


def check_image(
    image_path: Path,
    *,
    target_lang: str,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> ImageCheckRecord:
    """Ask the vision LLM whether ``image_path`` contains ``target_lang`` text.

    Single-image entry point. The caller is responsible for verifying
    that ``vision_model`` supports vision (the directory-scanning
    :func:`check_images` does that up front, so a per-image call from
    inside a batch never repeats the check).
    """
    data_url = _encode_image(image_path)
    user_prompt = (
        f"Does this image contain any visible text written in "
        f"'{target_lang}'? If the target language is missing or you "
        "cannot tell, set contains to false."
    )
    response = litellm.completion(
        model=vision_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        max_tokens=_MAX_TOKENS,
    )
    raw = response.choices[0].message.content or "{}"
    contains, reason = _parse_response(raw)
    return ImageCheckRecord(
        path=str(image_path),
        contains_target_lang=contains,
        reason=reason,
    )


def _supports_vision(model: str) -> bool:
    """Return False when ``litellm`` reports the model as non-vision OR
    when the lookup itself raised; return True only when ``litellm``
    explicitly confirms vision support or the helper is missing.

    The wrapper distinguishes "lookup is unavailable on this install"
    from "lookup ran and failed":

    * ``litellm.supports_vision`` does not exist (older ``litellm``
      versions; the project's ``pyproject.toml`` still declares
      ``litellm>=1.0.0`` and bumping that floor is out of scope for
      this CLI). Returns ``True`` so the operator can still reach the
      real completion call — failing closed would block every model
      on older deps the project still claims to support, and there is
      no evidence the model is non-vision.
    * The helper returns a truthy value. Returns ``True``.
    * The helper returns a falsy value. Returns ``False`` — explicit
      non-vision verdict; the gate surfaces the usage error with the
      override hint.
    * The helper exists but raises (unrecognised model, stale
      registry, etc.). Returns ``False`` — the contract is that
      unsupported / unrecognised vision models fail as a usage error
      before any completion call; a typo in ``--vision-model`` must
      not silently reach the API and burn tokens. Operators with a
      real-but-unregistered model can fall back to an older
      ``litellm`` that lacks the helper entirely (the missing-helper
      path above) or pin a recognised model string.
    """
    helper = getattr(litellm, "supports_vision", None)
    if helper is None:
        return True
    try:
        return bool(helper(model=model))
    except Exception:
        return False


def check_images(
    image_or_dir: Path,
    *,
    target_lang: str,
    vision_model: str = DEFAULT_VISION_MODEL,
) -> List[ImageCheckRecord]:
    """Check a single image or every image under a directory.

    Validates the input path AND the vision capability of
    ``vision_model`` up front so a typo in ``--vision-model`` fails
    before any tokens are billed. Per-image LLM calls are sequential —
    the typical operator runs this on a small screenshot bundle, and
    parallelism would complicate rate-limit handling without a
    real-world win at this scale.
    """
    if not image_or_dir.exists():
        raise FileNotFoundError(
            f"image path does not exist: {image_or_dir}"
        )
    if image_or_dir.is_file():
        if image_or_dir.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(
                f"image file has unsupported extension: {image_or_dir} "
                f"(supported: {' '.join(IMAGE_EXTENSIONS)})"
            )
    elif not image_or_dir.is_dir():
        raise ValueError(
            f"image path is neither a file nor a directory: {image_or_dir}"
        )

    if not _supports_vision(vision_model):
        raise ValueError(
            f"vision model not supported: {vision_model!r}. Pick a "
            "vision-capable model via --vision-model (default: "
            f"{DEFAULT_VISION_MODEL!r})."
        )

    images = _iter_images(image_or_dir)
    return [
        check_image(p, target_lang=target_lang, vision_model=vision_model)
        for p in images
    ]


def add_check_image_subparser(
    sub: "argparse._SubParsersAction[argparse.ArgumentParser]",
) -> argparse.ArgumentParser:
    """Register the ``check-image`` subcommand on the top-level argparse.

    Kept as a public helper so :mod:`mdpo_llm.__main__` can attach the
    subparser without importing implementation symbols individually,
    matching :func:`add_lint_subparser` / :func:`add_validate_dir_subparser`.
    """
    p = sub.add_parser(
        "check-image",
        help=(
            "Vision-LLM scan: flag images that still contain visible "
            "text in --target (typically the source language of the "
            "translation, used to surface un-localised screenshots)."
        ),
        description=(
            "Walk a single image or a directory of images and ask a "
            "vision-capable LLM whether each image contains visible "
            "text written in --target. Emits a JSON array of {path, "
            "contains_target_lang, reason} records to stdout. The verb "
            "exists to catch screenshots whose UI text was localized in "
            "code but whose image asset still shows the source-locale "
            "rendering — pass the SOURCE language of the translation as "
            "--target and treat 'contains_target_lang=true' records as "
            "residue. This class of finding is invisible to the "
            "text-only validators (mdpo-llm lint, the T-16 LLM grader, "
            "the T-17 residue post-pass)."
        ),
    )
    p.add_argument(
        "image_path",
        help=(
            "Path to a single image file or a directory of images "
            "(scanned recursively). Supported extensions: "
            + " ".join(IMAGE_EXTENSIONS)
            + "."
        ),
    )
    p.add_argument(
        "--target",
        required=True,
        help=(
            "BCP 47 locale of the language the vision LLM should look "
            "for in each image. For the residue workflow this is the "
            "SOURCE language of the translation (e.g. 'en' when "
            "scanning an English→Korean translated tree's "
            "screenshots): records with 'contains_target_lang=true' "
            "then carry un-localised source-language text and are the "
            "findings the verb is meant to surface."
        ),
    )
    p.add_argument(
        "--vision-model",
        default=DEFAULT_VISION_MODEL,
        help=(
            "Vision-capable LiteLLM model string (default: "
            f"{DEFAULT_VISION_MODEL!r}). Validated via "
            "`litellm.supports_vision` before any API call so a "
            "non-vision model surfaces as a usage error rather than "
            "burning tokens on a model that cannot read images."
        ),
    )
    p.add_argument(
        "--exit-non-zero-on-findings",
        dest="exit_non_zero_on_findings",
        action="store_true",
        help=(
            "Exit with code 1 when any image is flagged "
            "(contains_target_lang=true). Default: always exit 0 "
            "unless a usage error occurs. Use this in CI to fail the "
            "build on un-localised image assets."
        ),
    )
    p.set_defaults(func=cmd_check_image)
    return p


def cmd_check_image(args: argparse.Namespace) -> int:
    """``mdpo-llm check-image`` entry point.

    Exit code contract:
      * ``2`` — usage error (missing path, unsupported extension on a
        single-file argument, non-vision ``--vision-model``).
        Surfaced before any API call.
      * ``1`` — at least one image flagged AND
        ``--exit-non-zero-on-findings`` was passed.
      * ``0`` — otherwise (no findings, or findings without the opt-in
        flag).
    """
    image_path = Path(args.image_path)
    try:
        records = check_images(
            image_path,
            target_lang=args.target,
            vision_model=args.vision_model,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    payload = [r.to_dict() for r in records]
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if getattr(args, "exit_non_zero_on_findings", False) and any(
        r.contains_target_lang for r in records
    ):
        return 1
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Standalone entry point: ``python -m mdpo_llm.cli_check_image …``.

    Mirrors the ``mdpo-llm check-image`` subcommand surface so the
    module can be driven directly without going through the top-level
    parser — convenient for ad-hoc runs and for tests that exercise
    the CLI surface in isolation.
    """
    parser = argparse.ArgumentParser(prog="mdpo-llm-check-image")
    sub = parser.add_subparsers(dest="command", required=True)
    add_check_image_subparser(sub)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
