"""Tests for the vision-LLM image residue scanner ``mdpo-llm check-image`` (T-22)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mdpo_llm.cli_check_image import (
    DEFAULT_VISION_MODEL,
    IMAGE_EXTENSIONS,
    ImageCheckRecord,
    SYSTEM_PROMPT,
    add_check_image_subparser,
    check_image,
    check_images,
    cmd_check_image,
    main,
)


# ---------------------------------------------------------------------------
# Test helpers.
# ---------------------------------------------------------------------------


# A 1x1 transparent PNG. Real bytes so MIME sniffers and base64 encoders
# behave the same as on a real screenshot; vision LLMs reject zero-byte
# payloads, so the mock fixture also pretends to "see" this content
# rather than just trusting the wire payload was non-empty.
_PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4"
    b"\x89\x00\x00\x00\rIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def _write_image(path: Path, body: bytes = _PNG_1X1) -> Path:
    """Write a synthetic image file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return path


def _mock_response(contains: bool, reason: str = "stub reason") -> MagicMock:
    """Build a litellm-shaped completion response carrying a JSON payload."""
    response = MagicMock()
    response.choices[0].message.content = json.dumps(
        {"contains": contains, "reason": reason}
    )
    return response


# ---------------------------------------------------------------------------
# Single-image positive / negative paths.
# ---------------------------------------------------------------------------


class TestSingleImage:
    def test_positive_response_records_finding(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(
                True, "Korean text visible in banner"
            )
            records = check_images(img, target_lang="ko", vision_model="vmodel")
        assert len(records) == 1
        assert records[0].contains_target_lang is True
        assert "Korean" in records[0].reason
        assert records[0].path == str(img)

    def test_negative_response_records_clean(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "clean.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(
                False, "No target-language text"
            )
            records = check_images(img, target_lang="ko", vision_model="vmodel")
        assert len(records) == 1
        assert records[0].contains_target_lang is False
        assert records[0].reason == "No target-language text"

    def test_single_image_invokes_litellm_with_strict_prompt(
        self, tmp_path: Path
    ) -> None:
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(False)
            check_image(img, target_lang="ja", vision_model="vmodel")
        assert mock_litellm.completion.call_count == 1
        kwargs = mock_litellm.completion.call_args.kwargs
        assert kwargs["model"] == "vmodel"
        # The system prompt must be the strict OCR prompt verbatim so
        # the two implementations (mdpo-llm + ) stay
        # decision-aligned across forks.
        assert kwargs["messages"][0]["role"] == "system"
        assert kwargs["messages"][0]["content"] == SYSTEM_PROMPT
        user_content = kwargs["messages"][1]["content"]
        assert any(part.get("type") == "image_url" for part in user_content)
        # The image is wired as a data: URL with a real MIME type so
        # litellm routes it through the vision path of every supported
        # provider.
        image_url = next(
            part["image_url"]["url"]
            for part in user_content
            if part.get("type") == "image_url"
        )
        assert image_url.startswith("data:image/png;base64,")

    def test_response_with_code_fence_is_parsed(self, tmp_path: Path) -> None:
        # Models sometimes wrap their JSON in a ```json fence even when
        # the system prompt forbids it. The parser strips the fence
        # rather than dropping the entire response.
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            response = MagicMock()
            response.choices[0].message.content = (
                '```json\n{"contains": true, "reason": "fenced"}\n```'
            )
            mock_litellm.completion.return_value = response
            records = check_images(img, target_lang="ko", vision_model="vmodel")
        assert records[0].contains_target_lang is True
        assert records[0].reason == "fenced"

    def test_invalid_json_response_surfaces_as_value_error(
        self, tmp_path: Path
    ) -> None:
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            response = MagicMock()
            response.choices[0].message.content = "not json at all"
            mock_litellm.completion.return_value = response
            with pytest.raises(ValueError, match="valid JSON"):
                check_images(img, target_lang="ko", vision_model="vmodel")

    def test_string_false_does_not_coerce_to_true(
        self, tmp_path: Path
    ) -> None:
        # Some providers emit ``"contains": "false"`` instead of a JSON
        # boolean. A naked ``bool()`` on the non-empty string would
        # return True and silently fail a CI run under
        # ``--exit-non-zero-on-findings``. The parser normalises
        # recognised string aliases instead.
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            response = MagicMock()
            response.choices[0].message.content = (
                '{"contains": "false", "reason": "stringified bool"}'
            )
            mock_litellm.completion.return_value = response
            records = check_images(
                img, target_lang="ko", vision_model="vmodel"
            )
        assert records[0].contains_target_lang is False

    def test_string_true_aliases_accepted(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "screen.png")
        for alias in ("true", "yes", "1", "TRUE", "YES"):
            with patch(
                "mdpo_llm.cli_check_image.litellm"
            ) as mock_litellm:
                mock_litellm.supports_vision.return_value = True
                response = MagicMock()
                response.choices[0].message.content = json.dumps(
                    {"contains": alias, "reason": alias}
                )
                mock_litellm.completion.return_value = response
                records = check_images(
                    img, target_lang="ko", vision_model="vmodel"
                )
            assert records[0].contains_target_lang is True, alias

    def test_unrecognised_contains_value_raises(self, tmp_path: Path) -> None:
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            response = MagicMock()
            response.choices[0].message.content = (
                '{"contains": "maybe", "reason": "unclear"}'
            )
            mock_litellm.completion.return_value = response
            with pytest.raises(ValueError, match="must be a JSON boolean"):
                check_images(
                    img, target_lang="ko", vision_model="vmodel"
                )

    def test_missing_contains_field_raises(self, tmp_path: Path) -> None:
        # An empty / partially-formed response (``{}`` or
        # ``{"reason": "..."}`` with no ``contains``) MUST NOT default
        # to a clean verdict — under --exit-non-zero-on-findings that
        # would silently mask provider outages and pass a broken CI
        # run as green.
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            response = MagicMock()
            response.choices[0].message.content = '{"reason": "no verdict"}'
            mock_litellm.completion.return_value = response
            with pytest.raises(ValueError, match="missing the required"):
                check_images(
                    img, target_lang="ko", vision_model="vmodel"
                )

    def test_empty_json_object_response_raises(self, tmp_path: Path) -> None:
        # The provider may emit an empty object when it refuses to
        # answer (safety filter, token limit). The parser must reject
        # rather than fabricate a False verdict.
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            response = MagicMock()
            response.choices[0].message.content = "{}"
            mock_litellm.completion.return_value = response
            with pytest.raises(ValueError, match="missing the required"):
                check_images(
                    img, target_lang="ko", vision_model="vmodel"
                )


# ---------------------------------------------------------------------------
# Directory scan.
# ---------------------------------------------------------------------------


class TestDirectoryScan:
    def test_directory_scan_visits_every_image(self, tmp_path: Path) -> None:
        _write_image(tmp_path / "a.png")
        _write_image(tmp_path / "nested" / "b.jpg")
        _write_image(tmp_path / "c.gif")
        # Non-image files must be ignored even when sandwiched between
        # real images.
        (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(False)
            records = check_images(
                tmp_path, target_lang="ko", vision_model="vmodel"
            )
        assert {Path(r.path).name for r in records} == {"a.png", "b.jpg", "c.gif"}
        # One LLM call per image — the verb is sequential by design.
        assert mock_litellm.completion.call_count == 3

    def test_directory_scan_records_findings_per_image(
        self, tmp_path: Path
    ) -> None:
        _write_image(tmp_path / "dirty.png")
        _write_image(tmp_path / "clean.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            # The directory walk is sorted, so ``clean.png`` is visited
            # before ``dirty.png``. Pair the side_effect responses to
            # that order so each record carries the right verdict.
            mock_litellm.completion.side_effect = [
                _mock_response(False, "no target text"),
                _mock_response(True, "still has Korean"),
            ]
            records = check_images(
                tmp_path, target_lang="ko", vision_model="vmodel"
            )
        records_by_name = {Path(r.path).name: r for r in records}
        assert records_by_name["clean.png"].contains_target_lang is False
        assert records_by_name["dirty.png"].contains_target_lang is True

    def test_empty_directory_returns_empty_list(self, tmp_path: Path) -> None:
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(False)
            records = check_images(
                tmp_path, target_lang="ko", vision_model="vmodel"
            )
        assert records == []
        assert mock_litellm.completion.call_count == 0

    def test_directory_walk_is_sorted_for_stable_output(
        self, tmp_path: Path
    ) -> None:
        # The output JSON list must be byte-stable across runs and
        # filesystems with different walk order; sort by path here
        # locks that contract in.
        _write_image(tmp_path / "z.png")
        _write_image(tmp_path / "a.png")
        _write_image(tmp_path / "m.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(False)
            records = check_images(
                tmp_path, target_lang="ko", vision_model="vmodel"
            )
        names = [Path(r.path).name for r in records]
        assert names == sorted(names)


# ---------------------------------------------------------------------------
# Vision-model gating.
# ---------------------------------------------------------------------------


class TestVisionModelGate:
    def test_non_vision_model_raises_before_any_completion(
        self, tmp_path: Path
    ) -> None:
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = False
            with pytest.raises(ValueError, match="vision model not supported"):
                check_images(
                    img, target_lang="ko", vision_model="not-vision"
                )
            # The brief mandates: never burn tokens on a non-vision
            # model. The gate runs before any completion call.
            assert mock_litellm.completion.call_count == 0

    def test_supports_vision_exception_blocks_run(
        self, tmp_path: Path
    ) -> None:
        # An existing-but-raising helper means the registry lookup
        # actually ran and failed (unrecognised model, stale registry,
        # etc.). The CLI contract is that unsupported / unrecognised
        # vision models fail as a usage error BEFORE any completion
        # call so a typo in --vision-model cannot reach the API and
        # burn tokens.
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.side_effect = RuntimeError(
                "model not in registry"
            )
            with pytest.raises(ValueError, match="vision model not supported"):
                check_images(
                    img, target_lang="ko", vision_model="mystery"
                )
            assert mock_litellm.completion.call_count == 0

    def test_missing_supports_vision_helper_falls_through(
        self, tmp_path: Path
    ) -> None:
        # Older litellm releases lack ``supports_vision`` entirely.
        # ``pyproject.toml`` still declares ``litellm>=1.0.0`` (out of
        # scope to bump from this CLI), so the gate must degrade to
        # "let the completion call surface any real failure" rather
        # than blocking every model.
        img = _write_image(tmp_path / "screen.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            # ``del`` removes the attribute from the MagicMock so
            # getattr() returns the default sentinel — emulating an
            # older litellm install.
            del mock_litellm.supports_vision
            mock_litellm.completion.return_value = _mock_response(False)
            records = check_images(
                img, target_lang="ko", vision_model="legacy-vision-model"
            )
        assert len(records) == 1
        assert mock_litellm.completion.call_count == 1


# ---------------------------------------------------------------------------
# Path validation.
# ---------------------------------------------------------------------------


class TestPathValidation:
    def test_missing_path_raises_file_not_found(self, tmp_path: Path) -> None:
        with patch("mdpo_llm.cli_check_image.litellm"):
            with pytest.raises(FileNotFoundError, match="does not exist"):
                check_images(
                    tmp_path / "missing.png",
                    target_lang="ko",
                    vision_model="vmodel",
                )

    def test_single_file_with_unsupported_extension_raises(
        self, tmp_path: Path
    ) -> None:
        bad = tmp_path / "doc.txt"
        bad.write_text("not an image", encoding="utf-8")
        with patch("mdpo_llm.cli_check_image.litellm"):
            with pytest.raises(ValueError, match="unsupported extension"):
                check_images(
                    bad, target_lang="ko", vision_model="vmodel"
                )


# ---------------------------------------------------------------------------
# JSON output schema + CLI exit-code contract.
# ---------------------------------------------------------------------------


class TestJsonOutputSchema:
    def test_cli_emits_json_array_of_records(
        self, tmp_path: Path, capsys
    ) -> None:
        _write_image(tmp_path / "a.png")
        _write_image(tmp_path / "b.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.side_effect = [
                _mock_response(True, "korean banner"),
                _mock_response(False, "clean"),
            ]
            rc = main([
                "check-image",
                str(tmp_path),
                "--target",
                "ko",
                "--vision-model",
                "vmodel",
            ])
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert len(payload) == 2
        # Schema MUST be exactly the documented three keys per record;
        # downstream CI parsers depend on this contract.
        assert set(payload[0].keys()) == {
            "path",
            "contains_target_lang",
            "reason",
        }

    def test_record_to_dict_is_json_serialisable(self) -> None:
        record = ImageCheckRecord(
            path="screen.png",
            contains_target_lang=True,
            reason="found",
        )
        encoded = json.dumps(record.to_dict())
        decoded = json.loads(encoded)
        assert decoded == {
            "path": "screen.png",
            "contains_target_lang": True,
            "reason": "found",
        }


class TestExitCodes:
    def test_no_findings_exits_zero(
        self, tmp_path: Path, capsys
    ) -> None:
        _write_image(tmp_path / "a.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(False)
            rc = main([
                "check-image",
                str(tmp_path),
                "--target",
                "ko",
                "--vision-model",
                "vmodel",
            ])
        assert rc == 0

    def test_findings_default_exits_zero(
        self, tmp_path: Path, capsys
    ) -> None:
        # Without --exit-non-zero-on-findings, findings are reported
        # but the command still exits 0 — matches the lint /
        # validate-dir verb contract.
        _write_image(tmp_path / "a.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(True)
            rc = main([
                "check-image",
                str(tmp_path),
                "--target",
                "ko",
                "--vision-model",
                "vmodel",
            ])
        assert rc == 0

    def test_findings_exit_non_zero_when_flag_set(
        self, tmp_path: Path, capsys
    ) -> None:
        _write_image(tmp_path / "a.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = True
            mock_litellm.completion.return_value = _mock_response(True)
            rc = main([
                "check-image",
                str(tmp_path),
                "--target",
                "ko",
                "--vision-model",
                "vmodel",
                "--exit-non-zero-on-findings",
            ])
        assert rc == 1

    def test_missing_path_exits_two(self, tmp_path: Path, capsys) -> None:
        with patch("mdpo_llm.cli_check_image.litellm"):
            rc = main([
                "check-image",
                str(tmp_path / "nope.png"),
                "--target",
                "ko",
                "--vision-model",
                "vmodel",
            ])
        assert rc == 2
        assert "does not exist" in capsys.readouterr().err

    def test_non_vision_model_exits_two(
        self, tmp_path: Path, capsys
    ) -> None:
        _write_image(tmp_path / "a.png")
        with patch("mdpo_llm.cli_check_image.litellm") as mock_litellm:
            mock_litellm.supports_vision.return_value = False
            rc = main([
                "check-image",
                str(tmp_path),
                "--target",
                "ko",
                "--vision-model",
                "definitely-not-vision",
            ])
        assert rc == 2
        captured = capsys.readouterr()
        assert "vision model not supported" in captured.err

    def test_default_vision_model_is_brief_default(self) -> None:
        # Pin the default so a careless change to the constant trips a
        # test: the brief explicitly names this model and CHANGELOG
        # entries reference it.
        assert DEFAULT_VISION_MODEL == "openrouter/openai/gpt-4o"


# ---------------------------------------------------------------------------
# Subcommand wiring.
# ---------------------------------------------------------------------------


class TestSubparserWiring:
    def test_add_check_image_subparser_registers_func(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command", required=True)
        add_check_image_subparser(sub)
        args = parser.parse_args([
            "check-image",
            "some-path",
            "--target",
            "ko",
        ])
        assert args.func is cmd_check_image
        # Default vision model must come through argparse defaults so
        # bare ``check-image PATH --target LANG`` works without
        # specifying a model explicitly.
        assert args.vision_model == DEFAULT_VISION_MODEL

    def test_supported_extensions_advertised_in_help(self) -> None:
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command", required=True)
        p = add_check_image_subparser(sub)
        help_text = p.format_help()
        for ext in IMAGE_EXTENSIONS:
            assert ext in help_text
