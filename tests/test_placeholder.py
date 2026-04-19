"""Unit tests for the placeholder substitution framework (T-4)."""

from __future__ import annotations

import re

import pytest

from mdpo_llm.placeholder import (
    ANCHOR_PATTERN,
    BUILTIN_PATTERNS,
    HTML_ATTR_PATTERN,
    Placeholder,
    PlaceholderMap,
    PlaceholderRegistry,
    TOKEN_RE,
    check_round_trip,
    check_structural_position,
    format_token,
)
from mdpo_llm.validator import validate


# ---------- format_token / TOKEN_RE ----------


def test_format_token_produces_expected_shape():
    assert format_token(0) == "\u27e6P:0\u27e7"
    assert format_token(12) == "\u27e6P:12\u27e7"


def test_token_re_matches_format_token_output():
    for i in (0, 1, 99, 1000):
        m = TOKEN_RE.fullmatch(format_token(i))
        assert m is not None
        assert int(m.group(1)) == i


# ---------- empty / no-match behaviour ----------


def test_empty_registry_is_passthrough():
    reg = PlaceholderRegistry()
    text = "See https://example.com for details."
    encoded, mapping = reg.encode(text)
    assert encoded == text
    assert len(mapping) == 0
    assert bool(mapping) is False
    assert reg.decode(encoded, mapping) == text


def test_registry_with_no_matches_returns_empty_map():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    text = "Nothing to replace here."
    encoded, mapping = reg.encode(text)
    assert encoded == text
    assert len(mapping) == 0


# ---------- encode basics ----------


def test_encode_replaces_all_non_overlapping_matches():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    text = "See https://a.example and https://b.example for docs."
    encoded, mapping = reg.encode(text)

    assert len(mapping) == 2
    tokens = mapping.tokens()
    assert tokens == ["\u27e6P:0\u27e7", "\u27e6P:1\u27e7"]
    assert encoded == f"See {tokens[0]} and {tokens[1]} for docs."
    originals = [p.original for p in mapping]
    assert originals == ["https://a.example", "https://b.example"]
    assert all(p.pattern_name == "url" for p in mapping)


def test_encode_indexes_in_source_order():
    reg = PlaceholderRegistry()
    reg.register("word", r"[A-Za-z]+")
    text = "alpha beta gamma"
    encoded, mapping = reg.encode(text)
    # Three matches, numbered 0/1/2 in source order.
    assert [p.token for p in mapping] == [format_token(i) for i in range(3)]
    assert [p.original for p in mapping] == ["alpha", "beta", "gamma"]


def test_encode_with_multiple_patterns_all_fire():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    reg.register("bracketed", r"\[[^\]]+\]")
    text = "[anchor] link https://example.com end"
    encoded, mapping = reg.encode(text)
    assert len(mapping) == 2
    # Earliest match first: the bracket starts at index 0.
    assert mapping.items[0].pattern_name == "bracketed"
    assert mapping.items[0].original == "[anchor]"
    assert mapping.items[1].pattern_name == "url"
    assert mapping.items[1].original == "https://example.com"


# ---------- overlap resolution ----------


def test_overlap_earlier_start_wins():
    reg = PlaceholderRegistry()
    reg.register("outer", r"abc")
    reg.register("inner", r"bc")
    text = "abc"
    encoded, mapping = reg.encode(text)
    # Both candidates start inside 'abc'; the outer starts first at index 0.
    assert len(mapping) == 1
    assert mapping.items[0].original == "abc"
    assert encoded == format_token(0)


def test_overlap_same_start_longest_wins():
    reg = PlaceholderRegistry()
    # Register short first so we know the tiebreaker isn't "registration order".
    reg.register("short", r"a")
    reg.register("long", r"abc")
    text = "abc"
    encoded, mapping = reg.encode(text)
    assert len(mapping) == 1
    assert mapping.items[0].original == "abc"
    assert mapping.items[0].pattern_name == "long"


def test_overlap_nested_inner_match_dropped():
    reg = PlaceholderRegistry()
    reg.register("outer", r"\[\[[^\]]+\]\]")
    reg.register("inner", r"\[[^\]]+\]")
    text = "[[outer]]"
    encoded, mapping = reg.encode(text)
    # The outer spans the whole string; the inner would match '[outer]'
    # but starts inside the consumed span and must be dropped.
    assert len(mapping) == 1
    assert mapping.items[0].original == "[[outer]]"


def test_zero_width_matches_are_skipped():
    reg = PlaceholderRegistry()
    reg.register("empty", r"")  # matches between every character
    text = "abc"
    encoded, mapping = reg.encode(text)
    assert encoded == text
    assert len(mapping) == 0


# ---------- decode / round-trip ----------


def test_encode_decode_is_identity():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    reg.register("code", r"`[^`\n]+`")
    text = "See `fn()` at https://example.com for `more`."
    encoded, mapping = reg.encode(text)
    assert reg.decode(encoded, mapping) == text


def test_decode_ignores_unknown_tokens():
    mapping = PlaceholderMap(
        items=[Placeholder(token=format_token(0), original="X", pattern_name="p")]
    )
    # ⟦P:5⟧ is not in the mapping; decode must preserve it verbatim so the
    # round-trip check can flag it as "unexpected".
    text = f"{format_token(0)} then {format_token(5)} end"
    decoded = PlaceholderRegistry.decode(text, mapping)
    assert decoded == f"X then {format_token(5)} end"


def test_decode_with_empty_mapping_returns_input():
    assert PlaceholderRegistry.decode("hello", PlaceholderMap()) == "hello"


# ---------- check_round_trip ----------


def test_round_trip_ok_when_all_tokens_present_exactly_once():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode("Go to https://a.example and https://b.example")
    translated = f"Vaya a {format_token(0)} y {format_token(1)}"
    assert check_round_trip(translated, mapping) is None


def test_round_trip_detects_missing_token():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode("See https://a.example")
    reason = check_round_trip("Translation without the token", mapping)
    assert reason is not None
    assert "missing=" in reason
    assert format_token(0) in reason


def test_round_trip_detects_duplicated_token():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode("See https://a.example")
    reason = check_round_trip(
        f"{format_token(0)} and again {format_token(0)}", mapping
    )
    assert reason is not None
    assert "duplicated=" in reason


def test_round_trip_detects_unexpected_token():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode("See https://a.example")
    # Input had one token (P:0); model fabricated P:7.
    reason = check_round_trip(
        f"{format_token(0)} plus hallucinated {format_token(7)}", mapping
    )
    assert reason is not None
    assert "unexpected=" in reason
    assert format_token(7) in reason


def test_round_trip_empty_mapping_always_ok():
    assert check_round_trip("whatever", PlaceholderMap()) is None
    assert check_round_trip("contains " + format_token(0), PlaceholderMap()) is None


def test_round_trip_combines_multiple_failures():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode(
        "A: https://a.example B: https://b.example C: https://c.example"
    )
    # P:0 missing, P:1 duplicated, P:9 unexpected.
    translated = (
        f"B={format_token(1)} "
        f"B2={format_token(1)} "
        f"C={format_token(2)} "
        f"junk={format_token(9)}"
    )
    reason = check_round_trip(translated, mapping)
    assert reason is not None
    assert "missing=" in reason
    assert "duplicated=" in reason
    assert "unexpected=" in reason


# ---------- pre-existing literal tokens in the source ----------


def test_preexisting_literal_token_is_recorded_and_preserved():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    # Source explains the placeholder syntax verbatim AND contains a URL
    # the registry would normally substitute.  Numbering must skip 0
    # because the literal already occupies that index.
    text = f"The token {format_token(0)} marks https://example.com anchors"
    encoded, mapping = reg.encode(text)
    # Literal stays put, URL becomes P:1 (P:0 is taken).
    assert format_token(0) in encoded
    assert format_token(1) in encoded
    assert "https://example.com" not in encoded
    assert [p.pattern_name for p in mapping] == ["__literal__", "url"]
    assert [p.token for p in mapping] == [format_token(0), format_token(1)]
    assert [p.original for p in mapping] == [format_token(0), "https://example.com"]


def test_preexisting_literal_survives_decode():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    text = f"Keep {format_token(0)} and see https://x.example"
    encoded, mapping = reg.encode(text)
    assert reg.decode(encoded, mapping) == text


def test_preexisting_literal_requires_preservation_in_output():
    # If the LLM drops a literal that was in the source, the round-trip
    # check must flag it — even though nothing in the registry produced it.
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode(
        f"See {format_token(0)} and https://a.example"
    )
    # Model forgot the literal.
    reason = check_round_trip(
        f"Vea {format_token(1)}", mapping
    )
    assert reason is not None
    assert "missing=" in reason
    assert format_token(0) in reason


def test_duplicate_literal_in_source_expects_two_in_output():
    # Two copies of the same literal token in the source → the round-trip
    # check must expect two copies in the output, not flag them as
    # duplicated.
    reg = PlaceholderRegistry()  # no patterns; only literals
    text = f"{format_token(0)} and {format_token(0)} again"
    encoded, mapping = reg.encode(text)
    assert encoded == text
    assert len(mapping) == 2
    # One copy in output → missing.
    reason_one = check_round_trip(f"only {format_token(0)}", mapping)
    assert reason_one is not None and "missing=" in reason_one
    # Two copies → OK.
    assert check_round_trip(text, mapping) is None
    # Three copies → duplicated.
    reason_three = check_round_trip(
        f"{format_token(0)} {format_token(0)} {format_token(0)}", mapping
    )
    assert reason_three is not None and "duplicated=" in reason_three


def test_registry_with_only_literals_still_records_them():
    # An empty registry but a source with literal tokens: encode still
    # produces a mapping so the round-trip check expects them verbatim.
    reg = PlaceholderRegistry()
    text = f"Literal {format_token(0)} here"
    encoded, mapping = reg.encode(text)
    assert encoded == text
    assert len(mapping) == 1
    assert mapping.items[0].pattern_name == "__literal__"


def test_pattern_match_overlapping_literal_is_dropped():
    # A pattern greedy enough to swallow a literal token must not win —
    # the literal's identity entry has priority by insertion/start order.
    reg = PlaceholderRegistry()
    reg.register("greedy", r"[^ ]+")  # would otherwise match the token
    text = f"word {format_token(0)} more"
    encoded, mapping = reg.encode(text)
    # "word" and "more" become tokens; the literal is preserved as-is.
    literals = [p for p in mapping if p.pattern_name == "__literal__"]
    assert len(literals) == 1
    assert literals[0].original == format_token(0)
    assert format_token(0) in encoded


def test_earlier_starting_pattern_cannot_absorb_literal():
    # Regression: a pattern whose match STARTS before a literal but spans
    # through it must be dropped, so the literal still gets its own
    # mapping entry and the round-trip check keeps guarding it.
    reg = PlaceholderRegistry()
    reg.register("link", r"\[[^\]]+\]\([^)]+\)")
    text = f"See [docs]({format_token(0)}) for more"
    encoded, mapping = reg.encode(text)

    # The literal gets its own entry (identity token).  The link is
    # dropped because it engulfs the literal's span.
    literals = [p for p in mapping if p.pattern_name == "__literal__"]
    links = [p for p in mapping if p.pattern_name == "link"]
    assert len(literals) == 1
    assert literals[0].original == format_token(0)
    assert len(links) == 0
    # Encoded text keeps the surrounding syntax intact because the
    # broader pattern was rejected.
    assert "[docs](" in encoded
    assert format_token(0) in encoded
    # Round-trip still requires the literal in the output.
    reason = check_round_trip("See [docs]() for more", mapping)
    assert reason is not None and "missing=" in reason


# ---------- registry API surface ----------


def test_register_accepts_precompiled_pattern():
    reg = PlaceholderRegistry()
    reg.register("url", re.compile(r"https?://\S+", re.IGNORECASE))
    encoded, mapping = reg.encode("See HTTPS://a.example/x")
    assert len(mapping) == 1
    assert mapping.items[0].original == "HTTPS://a.example/x"
    assert encoded == f"See {format_token(0)}"


def test_register_string_flags_applied():
    reg = PlaceholderRegistry()
    reg.register("word", r"[a-z]+", flags=re.IGNORECASE)
    _, mapping = reg.encode("HELLO world")
    assert [p.original for p in mapping] == ["HELLO", "world"]


def test_registry_bool_reflects_pattern_count():
    reg = PlaceholderRegistry()
    assert not reg
    reg.register("x", r"x")
    assert reg
    assert len(reg) == 1
    assert reg.patterns[0].name == "x"


def test_mapping_iter_yields_placeholders():
    reg = PlaceholderRegistry()
    reg.register("word", r"\w+")
    _, mapping = reg.encode("one two")
    tokens = [ph.token for ph in mapping]
    assert tokens == [format_token(0), format_token(1)]


# ---------- validator integration ----------


def test_validator_flags_missing_placeholder():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    source = "Visit https://example.com today"
    _, mapping = reg.encode(source)
    # Translation forgot to preserve the token.
    decoded_translation = "Visita hoy"
    result = validate(
        source,
        decoded_translation,
        target_lang="es",
        placeholder_map=mapping,
        encoded_translation=decoded_translation,  # still no token present
    )
    assert not result.ok
    assert any(i.check == "placeholder_roundtrip" for i in result.issues)


def test_validator_accepts_preserved_placeholders():
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    source = "Visit https://example.com today"
    _, mapping = reg.encode(source)
    encoded_translation = f"\uc624\ub298 {format_token(0)} \ubc29\ubb38"
    decoded_translation = "\uc624\ub298 https://example.com \ubc29\ubb38"
    result = validate(
        source,
        decoded_translation,
        target_lang="ko",
        placeholder_map=mapping,
        encoded_translation=encoded_translation,
    )
    assert result.ok, result.issues


def test_validator_runs_structural_checks_on_decoded_translation():
    # Regression for P2: a pattern that covers backticks (inline code)
    # must not break fence / inline-code counts when encoded text has
    # tokens in place of the syntax.  Structural checks MUST run on the
    # user-visible decoded translation.
    reg = PlaceholderRegistry()
    reg.register("inline_code", r"`[^`\n]+`")
    source = "Use `foo` and `bar`."
    encoded_src, mapping = reg.encode(source)
    # Encoded source: "Use ⟦P:0⟧ and ⟦P:1⟧." — no backticks survive.
    assert "`" not in encoded_src
    # The LLM returns an encoded response preserving both tokens.
    encoded_translation = (
        f"{format_token(0)}\uc640 {format_token(1)}\uc744 \uc0ac\uc6a9\ud558\uc138\uc694."
    )
    decoded_translation = "`foo`\uc640 `bar`\uc744 \uc0ac\uc6a9\ud558\uc138\uc694."
    result = validate(
        source,
        decoded_translation,
        target_lang="ko",
        mode="strict",  # exercise inline_code_count too
        placeholder_map=mapping,
        encoded_translation=encoded_translation,
    )
    assert result.ok, result.issues


def test_validator_without_placeholder_map_skips_roundtrip():
    # Regression guard: the new parameter must default to None so the
    # existing validator behaviour is preserved for callers that don't use
    # placeholders.
    result = validate(
        "Hello world", "\uc548\ub155 \uc138\uacc4", target_lang="ko"
    )
    assert result.ok


def test_validator_empty_mapping_is_noop():
    # An empty PlaceholderMap is passed when the registry matched nothing;
    # the validator must treat it identically to "no map".
    result = validate(
        "Hello world",
        "\uc548\ub155 \uc138\uacc4",
        target_lang="ko",
        placeholder_map=PlaceholderMap(),
    )
    assert result.ok


def test_validator_requires_encoded_translation_when_map_nonempty():
    # Regression guard: silently falling back to ``translation`` would
    # report every token as missing (decode strips them), so passing a
    # non-empty ``placeholder_map`` without ``encoded_translation`` must
    # raise rather than produce spurious validation failures.
    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")
    _, mapping = reg.encode("See https://a.example")
    with pytest.raises(ValueError, match="encoded_translation"):
        validate(
            "See https://a.example",
            "See https://a.example",
            target_lang="ko",
            placeholder_map=mapping,
        )


# ---------- processor wiring smoke test ----------


def test_processor_runs_placeholder_roundtrip_when_registry_passed():
    """End-to-end wiring: processor.MarkdownProcessor with a registry marks
    an entry fuzzy when the LLM drops a placeholder.

    Uses a stub ``_call_llm`` to avoid real API calls; the point is that
    the stash / decode / validator path fires when ``placeholders`` is
    configured and does nothing when it isn't.
    """
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=reg
    )
    proc.parser = MagicMock()
    proc.po_manager = MagicMock()
    proc.reconstructor = MagicMock()

    # Simulate ``_call_llm`` having just returned a translation that
    # dropped the placeholder.  Encode the source so the stash map is
    # non-empty.
    source = "See https://example.com"
    encoded, mapping = reg.encode(source)
    proc._tls.last_encoded_response = "See nothing"  # no token
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(entry, "See nothing", inplace=False, pool=MagicMock())
    assert ok is False
    assert "fuzzy" in entry.flags
    assert "placeholder_roundtrip" in (entry.tcomment or "")


def test_processor_no_registry_skips_roundtrip():
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    entry = polib.POEntry(msgid="hello", msgstr="")
    # No stash set; should return True (validation off + no placeholders).
    ok = proc._apply_validation(entry, "hello", inplace=False, pool=MagicMock())
    assert ok is True
    assert "fuzzy" not in entry.flags


def test_post_process_runs_before_decode_so_roundtrip_catches_rewrites():
    # Regression for cycle-4 P2: when placeholders are active, the
    # ``post_process`` hook must see the encoded LLM output.  If a hook
    # mangles a token (by accident or by overreach) the round-trip check
    # stashed by ``_call_llm`` must see the mangled text so validation
    # can flag it.  Running post-process on the decoded translation
    # would otherwise let the hook silently corrupt protected spans.
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    reg = PlaceholderRegistry()
    reg.register("url", r"https?://\S+")

    # Malicious (or buggy) hook that strips all ASCII-ish token glyphs.
    def strip_tokens(text: str) -> str:
        return TOKEN_RE.sub("", text)

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        placeholders=reg,
        post_process=strip_tokens,
    )

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content=f"{format_token(0)} \uc548\ub155"))
    ]
    fake_response.usage = None

    with patch("mdpo_llm.processor.litellm.completion", return_value=fake_response):
        proc._call_llm("See https://example.com")

    # The stash must reflect the post-processed (token-stripped) output
    # so the round-trip check can see the token went missing.
    assert format_token(0) not in proc._tls.last_encoded_response
    assert proc._tls.last_placeholder_map is not None
    reason = check_round_trip(
        proc._tls.last_encoded_response, proc._tls.last_placeholder_map
    )
    assert reason is not None and "missing=" in reason


def test_sequential_path_glossary_uses_unencoded_source():
    # Regression for cycle-3 P2: when placeholders are active, the
    # sequential ``_call_llm`` path must still derive the glossary block
    # from the raw source so a protected term (URL, anchor, etc.) that a
    # pattern replaced with a token still shows up in glossary context.
    # Otherwise sequential and batched paths would behave differently
    # for the same input.
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    reg = PlaceholderRegistry()
    # Pattern covers the whole URL — the glossary term ``example.com``
    # lives INSIDE the protected span.
    reg.register("url", r"https?://\S+")

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        placeholders=reg,
        glossary={"example.com": None},
        glossary_mode="instruction",
    )

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="ok"))]
    fake_response.usage = None
    with patch("mdpo_llm.processor.litellm.completion", return_value=fake_response) as mock:
        proc._call_llm("Visit https://example.com today")

    messages = mock.call_args.kwargs["messages"]
    system_content = messages[0]["content"]
    # Glossary must have picked up "example.com" from the raw source,
    # even though the user message shows the encoded (tokenized) form.
    assert "Glossary" in system_content
    assert "example.com" in system_content
    user_content = messages[-1]["content"]
    assert format_token(0) in user_content
    assert "https://example.com" not in user_content


# ---------- glossary placeholder mode ----------


def test_glossary_mode_defaults_to_placeholder():
    """Default mode substitutes glossary terms with ⟦P:N⟧ tokens pre-call
    and never leaks the raw terms into the system or user message."""
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None, "API": "API"},
    )
    assert proc.glossary_mode == "placeholder"

    captured = {}

    def _capture(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = None
        return resp

    with patch("mdpo_llm.processor.litellm.completion", side_effect=_capture):
        proc._call_llm("Visit GitHub to read the API docs")

    user = captured["messages"][-1]["content"]
    system = captured["messages"][0]["content"]
    assert "GitHub" not in user, (
        "placeholder mode must substitute terms out of the user message"
    )
    assert "\u27e6P:" in user, "placeholder mode must emit ⟦P:N⟧ tokens"
    assert "Glossary" not in system, (
        "placeholder mode must NOT append a glossary block to the system prompt"
    )


def test_glossary_mode_instruction_opt_in_restores_prompt_block():
    """Opt-in `instruction` mode keeps the legacy behaviour: glossary block in
    the system prompt, raw terms untouched in the user message."""
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None, "API": "API"},
        glossary_mode="instruction",
    )
    assert proc.glossary_mode == "instruction"

    captured = {}

    def _capture(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        resp.usage = None
        return resp

    with patch("mdpo_llm.processor.litellm.completion", side_effect=_capture):
        proc._call_llm("Visit GitHub to read the API docs")

    user = captured["messages"][-1]["content"]
    system = captured["messages"][0]["content"]
    assert "GitHub" in user, "instruction mode must NOT substitute terms"
    assert "API" in user
    assert "Glossary" in system
    assert "GitHub" in system


def test_glossary_placeholder_mode_substitutes_and_restores():
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None, "pull request": "\ud480 \ub9ac\ud018\uc2a4\ud2b8"},
        glossary_mode="placeholder",
    )

    captured = {}

    def _echo(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        user_text = kwargs["messages"][-1]["content"]
        resp = MagicMock()
        # LLM preserves tokens verbatim.
        resp.choices = [MagicMock(message=MagicMock(content=user_text))]
        resp.usage = None
        return resp

    with patch("mdpo_llm.processor.litellm.completion", side_effect=_echo):
        result = proc._call_llm("Visit GitHub to open a pull request today")

    user = captured["messages"][-1]["content"]
    system = captured["messages"][0]["content"]
    # Source text sent to the LLM has terms replaced with tokens.
    assert "GitHub" not in user
    assert "pull request" not in user
    assert format_token(0) in user or format_token(1) in user
    # Placeholder mode suppresses the instruction-mode glossary block so
    # the two paths don't double-feed terms to the model.
    assert "Glossary" not in system
    # Restored output: GitHub stays verbatim, "pull request" becomes the
    # target translation.
    assert "GitHub" in result
    assert "\ud480 \ub9ac\ud018\uc2a4\ud2b8" in result
    assert "pull request" not in result


def test_glossary_placeholder_mode_word_boundary_skips_morphology():
    # "APIs" (plural) must NOT match the glossary term "API" — the
    # trailing 's' is a word character and breaks the closing \b.  This is
    # the conservative false-negative rule: a missed match falls through
    # to the LLM, whereas a false-positive would corrupt neighbouring
    # text mid-word.
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"API": None},
        glossary_mode="placeholder",
    )

    captured = {}

    def _echo(*args, **kwargs):
        captured["messages"] = kwargs["messages"]
        user_text = kwargs["messages"][-1]["content"]
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=user_text))]
        resp.usage = None
        return resp

    with patch("mdpo_llm.processor.litellm.completion", side_effect=_echo):
        proc._call_llm("The APIs are documented but API is singular")

    user = captured["messages"][-1]["content"]
    # 'APIs' survives untouched; standalone 'API' becomes a token.
    assert "APIs" in user
    assert user.count("APIs") == 1
    # Exactly one bare-'API' match became a token.
    assert format_token(0) in user


def test_glossary_placeholder_mode_prefix_does_not_match_mid_word():
    # Mid-word occurrences must not be substituted either.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"git": None},
        glossary_mode="placeholder",
    )
    # "github" contains "git" but is a word on its own; word-boundary
    # should refuse to match.
    encoded, mapping = proc._encode_source("Use github and commit to git")
    assert "github" in encoded
    # One match for the standalone "git".
    glossary_entries = [p for p in mapping if p.pattern_name.startswith("glossary:")]
    assert len(glossary_entries) == 1
    assert glossary_entries[0].original == "git"


def test_glossary_placeholder_mode_longer_term_wins_on_overlap():
    # "pull request" and "pull" both match at the same start; the longer
    # term must win so "pull request" stays intact.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"pull": "P", "pull request": "PR"},
        glossary_mode="placeholder",
    )
    encoded, mapping = proc._encode_source("Open a pull request today")
    glossary_entries = [p for p in mapping if p.pattern_name.startswith("glossary:")]
    assert len(glossary_entries) == 1
    assert glossary_entries[0].original == "pull request"
    assert glossary_entries[0].replacement == "PR"


def test_glossary_placeholder_mode_skips_non_word_terms():
    # Terms starting or ending with non-word chars (e.g. ".NET", "C++")
    # cannot be matched reliably with \bterm\b — they are silently
    # skipped and fall through to the LLM's normal translation path.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={".NET": None, "C++": None, "API": None},
        glossary_mode="placeholder",
    )

    text = "Use API on .NET or C++"
    encoded, mapping = proc._encode_source(text)
    # Only "API" is substituted; ".NET" and "C++" are left alone.
    assert ".NET" in encoded
    assert "C++" in encoded
    assert "API" not in encoded
    glossary_entries = [p for p in mapping if p.pattern_name.startswith("glossary:")]
    assert len(glossary_entries) == 1
    assert glossary_entries[0].original == "API"


def test_glossary_placeholder_mode_do_not_translate_restored_verbatim():
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None},
        glossary_mode="placeholder",
    )
    encoded, mapping = proc._encode_source("Visit GitHub today")
    assert "GitHub" not in encoded
    # Decode with the SAME encoded text simulates the LLM echoing tokens
    # verbatim.  The restore must put the term back unchanged.
    decoded = PlaceholderRegistry.decode(encoded, mapping)
    assert decoded == "Visit GitHub today"


def test_glossary_placeholder_mode_translated_restores_target_form():
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"API": "\uc5d0\uc774\ud53c\uc544\uc774"},
        glossary_mode="placeholder",
    )
    encoded, mapping = proc._encode_source("Read the API docs")
    assert "API" not in encoded
    decoded = PlaceholderRegistry.decode(encoded, mapping)
    # Restored as the target-language form, not the original.
    assert "\uc5d0\uc774\ud53c\uc544\uc774" in decoded
    assert "API" not in decoded


def test_glossary_placeholder_mode_no_glossary_is_identity():
    # Placeholder mode without a glossary adds no glossary substitutions.
    # The T-6 built-ins still produce an (empty) mapping for text that
    # doesn't hit any built-in pattern — callers rely on ``bool(mapping)``
    # to tell an active encoding from a no-op.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary_mode="placeholder",
    )
    encoded, mapping = proc._encode_source("unchanged text")
    assert encoded == "unchanged text"
    assert mapping is not None
    assert len(mapping) == 0
    assert bool(mapping) is False


def test_glossary_placeholder_mode_combines_with_user_registry():
    # A user-supplied placeholder registry must keep working when glossary
    # placeholder mode adds its own patterns on top.
    from mdpo_llm.processor import MarkdownProcessor

    user_reg = PlaceholderRegistry()
    user_reg.register("url", r"https?://\S+")

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None},
        glossary_mode="placeholder",
        placeholders=user_reg,
    )
    encoded, mapping = proc._encode_source(
        "See https://example.com on GitHub"
    )
    # Both the URL (user pattern) and "GitHub" (glossary) become tokens.
    assert "https://example.com" not in encoded
    assert "GitHub" not in encoded
    # Decode restores the URL verbatim and GitHub (do-not-translate) too.
    decoded = PlaceholderRegistry.decode(encoded, mapping)
    assert decoded == "See https://example.com on GitHub"


def test_glossary_placeholder_mode_round_trip_catches_dropped_term():
    # If the LLM drops a glossary token the round-trip check must flag it
    # as missing, so _apply_validation marks the entry fuzzy.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None},
        glossary_mode="placeholder",
    )
    source = "Visit GitHub today"
    encoded, mapping = proc._encode_source(source)
    # Simulate a model response that forgot the token.
    proc._tls.last_encoded_response = "\ubc29\ubb38\ud558\uc138\uc694"
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(
        entry, "\ubc29\ubb38\ud558\uc138\uc694", inplace=False, pool=MagicMock()
    )
    assert ok is False
    assert "fuzzy" in entry.flags
    assert "placeholder_roundtrip" in (entry.tcomment or "")


def test_processor_structural_checks_run_on_decoded_not_encoded():
    # Regression for P2: when a placeholder covers inline code, the
    # encoded response has no backticks, but the decoded `processed`
    # text restores them.  Structural checks (strict-mode
    # inline_code_count) must look at decoded text and pass.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    reg = PlaceholderRegistry()
    reg.register("inline_code", r"`[^`\n]+`")

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        placeholders=reg,
        validation="strict",
    )

    source = "Use `foo` and `bar`."
    encoded_src, mapping = reg.encode(source)
    # LLM preserved both tokens verbatim in its (encoded) response.
    encoded_response = (
        f"{format_token(0)}\uc640 {format_token(1)}\uc744 \uc0ac\uc6a9\ud558\uc138\uc694."
    )
    decoded_processed = reg.decode(encoded_response, mapping)

    proc._tls.last_encoded_response = encoded_response
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    stats: dict = {"validated": 0, "validation_failed": 0}
    ok = proc._apply_validation(
        entry, decoded_processed, inplace=False, pool=MagicMock(), stats=stats
    )
    assert ok is True, entry.tcomment
    assert "fuzzy" not in entry.flags
    assert stats["validated"] == 1


# ---------- T-6 built-in patterns: anchor + inline HTML attributes ----------


def test_builtin_pattern_names_are_anchor_and_html_attr():
    # Defensive: downstream tools (CLI, introspection) key off these
    # pattern names when reporting which built-in fired.  Renaming either
    # silently is a breaking change, so pin the names in a dedicated test.
    assert [entry[0] for entry in BUILTIN_PATTERNS] == ["anchor", "html_attr"]


def test_anchor_pattern_matches_kramdown_ids():
    for text in (
        "## Overview {#overview}",
        "### Sub-section {#sub-section-1}",
        "Heading with underscores {#my_id}",
    ):
        m = ANCHOR_PATTERN.search(text)
        assert m is not None, text
        assert m.group(0).startswith("{#")
        assert m.group(0).endswith("}")


def test_anchor_pattern_rejects_non_anchor_braces():
    # ``{.cls}`` Kramdown class attributes and ``{key=val}`` styles are
    # NOT anchors; protecting them is out of scope for T-6.
    assert ANCHOR_PATTERN.search("text {.highlight}") is None
    assert ANCHOR_PATTERN.search("prose {key=val}") is None
    # Bare literal braces with no ``#`` are prose / code, not anchors.
    assert ANCHOR_PATTERN.search("dict {'a': 1}") is None


def test_anchor_not_protected_in_markdown_inline_code():
    # Regression for cycle-11 P1 from Codex: documents that discuss
    # anchor syntax with a literal example (``Use `{#overview}` on a
    # heading``) must NOT have the example frozen as an ``anchor``
    # placeholder — partial encoding would leave mixed
    # translated/untranslated text after decode and trigger spurious
    # round-trip failures when the model rewrites the example.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = "Use `{#overview}` on a heading line to define an anchor."
    encoded, mapping = proc._encode_source(text)
    anchor_hits = [p for p in mapping if p.pattern_name == "anchor"]
    assert anchor_hits == []


def test_anchor_protected_when_coexisting_with_inline_code():
    # Positive counterpart: a real anchor on a heading line must still
    # be protected even when the surrounding paragraph uses inline code.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = "## Overview {#overview}\n\nSee `some code` and `more`."
    encoded, mapping = proc._encode_source(text)
    anchor_hits = [p for p in mapping if p.pattern_name == "anchor"]
    assert [p.original for p in anchor_hits] == ["{#overview}"]


def test_anchor_pattern_matches_ial_with_extra_attrs():
    # Regression for cycle-2 P2 from Codex: Pandoc/Kramdown IAL syntax
    # allows the anchor id to co-appear with class / key=val attributes
    # inside the braces, e.g. ``{#overview .lead}``.  The pattern must
    # capture the full IAL so the id still round-trips through the LLM.
    for text in (
        "## Overview {#overview .lead}",
        "## Detail {#detail key=val}",
        "span text {#spec .highlight key=val}",
        "## Mixed {#mixed-id .cls1 .cls2}",
    ):
        m = ANCHOR_PATTERN.search(text)
        assert m is not None, text
        assert m.group(0).startswith("{#")
        assert m.group(0).endswith("}")
        # The anchor id must be inside the captured span.
        assert "#" in m.group(0)


def test_html_attr_pattern_matches_class_href_and_friends():
    text = '<a href="/docs" class="bare" data-id="42" xml:lang="en">link</a>'
    matches = HTML_ATTR_PATTERN.findall(text)
    # All four attribute pairs (quotes included) must be captured.
    assert 'href="/docs"' in matches
    assert 'class="bare"' in matches
    assert 'data-id="42"' in matches
    assert 'xml:lang="en"' in matches


def test_html_attr_pattern_handles_single_quotes():
    text = "<a href='/x' class='bare'>link</a>"
    matches = HTML_ATTR_PATTERN.findall(text)
    assert "href='/x'" in matches
    assert "class='bare'" in matches


def test_html_attr_pattern_skips_translatable_text_attrs():
    # Regression for cycle-1 P1 from Codex: attributes whose values are
    # user-facing prose (``alt``, ``title``, ``aria-label``,
    # ``placeholder``, ``label``) must NOT be captured — otherwise the
    # LLM never gets to translate accessibility / tooltip text and the
    # pipeline ships the source-language string unchanged.  The
    # allowlist approach intentionally skips these.
    text = (
        '<img src="/f.png" alt="Figure overview">'
        '<button title="Close the dialog" aria-label="Close">x</button>'
        '<input type="text" placeholder="Enter your name" label="Name">'
    )
    matches = HTML_ATTR_PATTERN.findall(text)
    # Identity / link attrs ARE captured.
    assert 'src="/f.png"' in matches
    assert 'type="text"' in matches
    # Translatable attrs are NOT captured.
    for localizable in ("alt=", "title=", "aria-label=", "placeholder=", "label="):
        assert not any(localizable in m for m in matches), (localizable, matches)


def test_html_attr_pattern_boundary_does_not_match_longer_names():
    # ``classified="x"`` must not be captured as ``class`` → with the
    # trailing ``\b`` the attribute-name alternation refuses to partially
    # match a longer identifier.
    text = '<div classified="true" srcxyz="no">'
    matches = HTML_ATTR_PATTERN.findall(text)
    assert matches == []


def test_html_attr_pattern_matches_srcset_not_just_src():
    # ``srcset`` is a distinct non-localizable attribute; must match in
    # full so responsive-image markup round-trips.
    text = '<img srcset="/a.png 1x, /b.png 2x" src="/a.png">'
    matches = HTML_ATTR_PATTERN.findall(text)
    assert 'srcset="/a.png 1x, /b.png 2x"' in matches
    assert 'src="/a.png"' in matches


def test_html_attr_pattern_matches_unquoted_values():
    # Regression for cycle-2 P2 from Codex: HTML5 allows unquoted
    # attribute values (no whitespace / quotes / angle brackets / `=`
    # inside).  The built-in must protect those too, otherwise
    # ``<img width=320 height=240>`` or ``<a href=/docs class=bare>``
    # slip past T-6 and the model can still mangle structural values.
    text = '<img width=320 height=240><a href=/docs class=bare>link</a>'
    matches = HTML_ATTR_PATTERN.findall(text)
    assert "width=320" in matches
    assert "height=240" in matches
    assert "href=/docs" in matches
    assert "class=bare" in matches


def test_html_attr_not_protected_in_markdown_inline_code():
    # Regression for cycle-10 P2 from Codex: HTML example text that
    # lives inside Markdown backticks (`` `<a href="/docs">` ``) or
    # inside a fenced code block must NOT be partially tokenized.
    # Partial protection — ``href`` becomes a token while ``<a`` / ``>``
    # stay as prose — would let the model rewrite the example while
    # the token count and structural checks all pass.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = 'Use `<a href="/docs">` to link to the guide.'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_html_attr_not_protected_inside_tilde_fence():
    # Regression for cycle-12 P2 from Codex: ``~~~`` fenced blocks are
    # supported by the Markdown parser used elsewhere and must be
    # treated as code by the built-in placeholder guard too.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = '~~~html\n<a href="/docs">link</a>\n~~~'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_html_attr_not_protected_in_multiline_backtick_span():
    # Regression for cycle-14 P2 from Codex: CommonMark allows inline
    # backtick spans to contain line breaks.  ``_find_code_ranges``
    # pairs runs globally rather than per-line so a backtick opening
    # on one line and closing on the next still hides the HTML example
    # from the placeholder layer.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = 'Use `<a href="/docs">\nin your markup` for links.'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_html_attr_not_protected_inside_multi_backtick_inline():
    # Regression for cycle-12 P2 from Codex: a multi-backtick inline
    # span (``\`\`...\`\```) that wraps content containing a single
    # backtick must also be treated as code.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = 'Use ``<a href="/docs">`` in the docs.'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_html_attr_not_protected_inside_indented_code_block():
    # Regression for cycle-15 P2 from Codex: CommonMark treats a line
    # starting with 4+ spaces (or a tab) as an indented code block.
    # The built-in placeholder guard must honour that too, otherwise
    # indented HTML examples get tokenized and can trip the new
    # round-trip / position validation.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = "Example:\n\n    <a href=\"/docs\">link</a>\n\nEnd."
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_anchor_not_protected_inside_indented_code_block():
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = "Example:\n\n    ## Title {#intro}\n\nEnd."
    encoded, mapping = proc._encode_source(text)
    anchor_hits = [p for p in mapping if p.pattern_name == "anchor"]
    assert anchor_hits == []


def test_indented_code_detected_without_blank_line_after_heading():
    # Regression for cycle-17 P2 from Codex: CommonMark allows an
    # indented code block to immediately follow a heading without a
    # blank line (headings aren't paragraphs, so the
    # "cannot-interrupt-a-paragraph" rule doesn't apply).  The built-in
    # guard must exempt that code too.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = "## Example\n    <a href=\"/docs\">link</a>"
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_html_attr_still_protected_in_list_item_continuation():
    # Regression for cycle-16 P2 from Codex: list-item continuation
    # lines are also 4-space indented but are NOT code.  The
    # indented-code heuristic defers to "not code" here so
    # structurally significant attributes in nested list content keep
    # their protection.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = (
        "- First item\n"
        '    <a href="/docs" class="bare">link</a> in the continuation.\n'
        "- Second item"
    )
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = sorted(
        p.original for p in mapping if p.pattern_name == "html_attr"
    )
    assert html_attr_hits == ['class="bare"', 'href="/docs"']


def test_anchor_still_protected_in_list_item_continuation():
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = (
        "- First item\n"
        "    Nested anchor {#pin-1} in a continuation line.\n"
        "- Second item"
    )
    encoded, mapping = proc._encode_source(text)
    anchor_hits = [p for p in mapping if p.pattern_name == "anchor"]
    assert [p.original for p in anchor_hits] == ["{#pin-1}"]


def test_html_attr_not_protected_in_list_item_code_block():
    # Regression for cycle-18 P2 from Codex: a blank line between
    # the list marker and an indented line marks a code block nested
    # under the list item (per CommonMark).  The built-in guard must
    # exempt those too.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = (
        "- item\n"
        "\n"
        '    <a href="/docs" class="bare">example</a>\n'
        "- next item"
    )
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_html_attr_not_protected_inside_blockquoted_tilde_fence():
    # Regression for cycle-18 P2 from Codex: fenced code blocks nested
    # inside a blockquote have the ``>`` marker before the fence.  The
    # fence regex accepts that prefix on both opener and closer.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = (
        "> ~~~html\n"
        '> <a href="/docs" class="bare">link</a>\n'
        "> ~~~"
    )
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_anchor_not_protected_inside_tilde_fence():
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = "~~~markdown\n## Overview {#overview}\n~~~"
    encoded, mapping = proc._encode_source(text)
    anchor_hits = [p for p in mapping if p.pattern_name == "anchor"]
    assert anchor_hits == []


def test_html_attr_protected_when_tag_neighbours_inline_code():
    # Positive counterpart: inline code before / after a real HTML
    # tag must NOT accidentally suppress the tag's protection.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = 'See `code` and then <a href="/docs">link</a> and `more`.'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    originals = [p.original for p in html_attr_hits]
    # Two paired backticks before the tag → even count → outside code.
    # ``href="/docs"`` is protected.
    assert originals == ['href="/docs"']


def test_html_attr_not_protected_in_prose_or_inline_code():
    # Regression for cycle-4 P1 from Codex: ``HTML_ATTR_PATTERN`` alone
    # matches bare ``class="primary"`` / ``href=/docs`` substrings, so
    # prose that discusses HTML would otherwise have its example strings
    # frozen.  The processor's effective registry applies an
    # in-tag-only predicate so only attributes inside a ``<...>`` open
    # tag are substituted.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = (
        'Use `class="primary"` to style the button, and `href=/docs` to '
        "point at the guide."
    )
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []
    assert encoded == text


def test_html_attr_not_matched_inside_another_attrs_quoted_value():
    # Regression for cycle-6 P2 from Codex: ``href="/docs"`` text
    # nested inside a translatable ``title`` value must NOT be frozen
    # as an ``html_attr`` placeholder — it's prose inside ``title``,
    # which the LLM still needs to translate.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = '<a title=\'see href="/docs"\' href="/real">link</a>'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    originals = sorted(p.original for p in html_attr_hits)
    # Only the REAL (top-level) href should be captured.
    assert originals == ['href="/real"']


def test_html_attr_protected_when_quoted_value_contains_gt():
    # Regression for cycle-5 P1 from Codex: a naive ``rfind("<")`` /
    # ``rfind(">")`` predicate terminates at the ``>`` inside ``"1 > 0"``
    # and leaves ``href``/later attrs unprotected.  The quote-aware tag
    # regex must keep those attributes inside the tag.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = '<a title="1 > 0" href="/docs" class="bare">link</a>'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    originals = sorted(p.original for p in html_attr_hits)
    # title is NOT on the allowlist (translatable), but href / class are
    # — and both must survive the ``>``-in-quote hazard.
    assert 'href="/docs"' in originals
    assert 'class="bare"' in originals


def test_html_attr_still_protected_inside_real_tags():
    # Positive counterpart to the predicate test: inside a real open
    # tag the built-in continues to substitute every allowlisted attr.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = '<a href="/docs" class="primary">link</a>'
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert len(html_attr_hits) == 2
    assert sorted(p.original for p in html_attr_hits) == sorted(
        ['href="/docs"', 'class="primary"']
    )


def test_html_attr_pattern_is_case_insensitive():
    # Regression for cycle-2 P3 from Codex: HTML attribute names are
    # case-insensitive, so the built-in must recognise ``HREF`` / ``Class``
    # in addition to the common lowercase form.  Otherwise uppercase
    # raw-HTML gets no protection.
    text = '<A HREF="/docs" Class="bare" XML:LANG="en">link</A>'
    matches = HTML_ATTR_PATTERN.findall(text)
    assert 'HREF="/docs"' in matches
    assert 'Class="bare"' in matches
    assert 'XML:LANG="en"' in matches


def test_builtin_patterns_are_always_active_without_user_registry():
    # Built-ins fire on every processor regardless of the ``placeholders``
    # kwarg — the T-6 brief explicitly forbids an opt-out flag.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = 'See the [API]{#api-ref} section via <span class="bare">widget</span>.'
    encoded, mapping = proc._encode_source(text)
    assert "{#api-ref}" not in encoded
    assert 'class="bare"' not in encoded
    # One anchor + one attribute = two tokens.
    assert len(mapping) == 2
    names = sorted(p.pattern_name for p in mapping)
    assert names == ["anchor", "html_attr"]


def test_five_anchors_and_three_class_attr_links_round_trip(tmp_path):
    # Regression fixture called out in the T-6 acceptance criteria: a
    # source block with 5 anchors + 3 class-attr HTML links must
    # round-trip every placeholder unchanged.  End-to-end: encode,
    # simulate an LLM that translates only the surrounding prose while
    # copying tokens through verbatim, decode, check the document
    # reconstructs with every anchor / attribute intact.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    source_block = (
        "Intro prose pointing at "
        '<a href="/a" class="bare">A</a>, '
        '<a href="/b" class="bare">B</a>, and '
        '<a href="/c" class="bare">C</a>. '
        "See also {#first-anchor} {#second-anchor} "
        "{#third-anchor} {#fourth-anchor} {#fifth-anchor}."
    )

    encoded, mapping = proc._encode_source(source_block)

    # 5 anchors + 3 links × 2 attrs (href + class) = 11 tokens.
    anchor_hits = [p for p in mapping if p.pattern_name == "anchor"]
    attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert len(anchor_hits) == 5
    assert len(attr_hits) == 6
    assert len(mapping) == 11

    # No protected content leaks into the encoded text.
    for protected in (
        "{#first-anchor}",
        "{#second-anchor}",
        "{#third-anchor}",
        "{#fourth-anchor}",
        "{#fifth-anchor}",
        'href="/a"',
        'href="/b"',
        'href="/c"',
        'class="bare"',
    ):
        assert protected not in encoded, protected

    # Simulate an LLM that echoes the encoded text (tokens intact) and
    # translates only the surrounding prose — the round-trip must pass.
    assert check_round_trip(encoded, mapping) is None

    # Decode restores every protected span byte-for-byte.
    decoded = PlaceholderRegistry.decode(encoded, mapping)
    assert decoded == source_block


def test_round_trip_flags_dropped_anchor_as_structural_fail():
    # The T-6 brief calls out "validator failure on a missing anchor is
    # a structural fail, not a warning."  A dropped anchor must surface
    # as a ``placeholder_roundtrip`` issue — the same hard-fail path
    # used by every placeholder pattern — regardless of validation mode.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    source = "## Overview {#overview}"
    encoded, mapping = proc._encode_source(source)
    assert len(mapping) == 1

    # Simulate an LLM response that dropped the anchor token.
    proc._tls.last_encoded_response = "## \uac1c\uc694"  # Korean for "Overview"
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(
        entry, "## \uac1c\uc694", inplace=False, pool=MagicMock()
    )
    assert ok is False
    assert "fuzzy" in entry.flags
    assert "placeholder_roundtrip" in (entry.tcomment or "")


def test_round_trip_flags_mangled_html_attr():
    # Same structural-fail contract for HTML attributes: if the LLM
    # rewrites ``class="bare"`` (even just reordering attributes or
    # translating the value), the token goes missing and the entry is
    # marked fuzzy.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    source = '<a href="/docs" class="bare">link</a>'
    encoded, mapping = proc._encode_source(source)
    # Both href and class produce tokens.
    assert len(mapping) == 2

    # Response drops the ``class="bare"`` token but keeps ``href="/docs"``.
    href_token = next(
        p.token for p in mapping if p.original == 'href="/docs"'
    )
    proc._tls.last_encoded_response = f'<a {href_token}>\ub9c1\ud06c</a>'
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(
        entry, "decoded-ignored", inplace=False, pool=MagicMock()
    )
    assert ok is False
    assert "fuzzy" in entry.flags


def test_structural_position_accepts_unchanged_structure():
    # Token-count-equal AND structurally-equal input/output should pass.
    source = "## Overview {#overview}\n\nSee <a href=\"/docs\">link</a>."
    # Decoded translation keeps anchor on the heading line and href in
    # the tag — just the prose changed.
    decoded = "## \uac1c\uc694 {#overview}\n\n<a href=\"/docs\">\ub9c1\ud06c</a>\ub97c \ucc38\uc870\ud558\uc138\uc694."
    assert check_structural_position(source, decoded) is None


def test_structural_position_flags_anchor_moved_off_heading():
    # Regression for cycle-3 P1 from Codex: a model that keeps the
    # anchor token count intact but relocates it off the heading into
    # the following paragraph passes round-trip but breaks document
    # structure.  The position check must flag it.
    source = "## Overview {#overview}\n\nIntro paragraph here."
    # Anchor slid off the heading into the paragraph.
    decoded = "## \uac1c\uc694\n\nIntro paragraph here. {#overview}"
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "anchor_heading" in reason


def test_structural_position_flags_attr_moved_outside_tag():
    # HTML attribute floating outside any ``<...>`` tag renders as
    # prose, not markup — this is a structural regression even when the
    # token round-trip passes.
    source = '<a href="/docs" class="bare">link</a>'
    # The ``class="bare"`` ended up outside the tag.
    decoded = '<a href="/docs">link</a> class="bare"'
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "html_attr_tag" in reason


def test_structural_position_flags_anchor_swap_between_headings():
    # Regression for cycle-4 P1 from Codex: count-based check missed
    # same-count relocations.  Per-ordinal multiset catches two anchors
    # swapping between two headings.
    source = "## Alpha {#a}\n\n## Beta {#b}"
    decoded = "## Alpha {#b}\n\n## Beta {#a}"  # swapped
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "anchor_heading_drift" in reason


def test_structural_position_flags_attr_mix_across_tags():
    # Regression for cycle-4 P1 from Codex: when attrs cross tag
    # boundaries, per-tag signature multiset must differ between source
    # and decoded.  Uses multi-attr tags so the class swap actually
    # changes the per-tag attr signatures — a single-attr swap between
    # structurally identical tags is intentionally undetectable and
    # that trade-off is acknowledged in :func:`_attr_tag_signatures`.
    source = '<a href="/a" class="x">A</a><a href="/b" class="y">B</a>'
    # Model swapped class between the two tags while keeping href in place.
    decoded = '<a href="/a" class="y">A</a><a href="/b" class="x">B</a>'
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "html_attr_tag_drift" in reason


def test_structural_position_ignores_attr_less_wrapper_tags():
    # Regression for cycle-7 P1 from Codex: a translation that wraps
    # the translated text in ``<strong>`` / ``<span>`` / ``<em>`` — or
    # unwraps one that was in the source — must NOT flag
    # ``html_attr_tag_drift``.  Only tags carrying a protected attribute
    # contribute to the signature multiset.
    source = '<a href="/docs">Read the <span>guide</span>.</a>'
    decoded = (
        '<a href="/docs"><strong>\uac00\uc774\ub4dc</strong>\ub97c '
        "\uc77d\uc73c\uc138\uc694.</a>"
    )
    # All protected attrs still round-trip; the accent wrappers change
    # but they have no attrs of their own.
    assert check_structural_position(source, decoded) is None


def test_structural_position_flags_attr_swap_across_different_tag_names():
    # Regression for cycle-9 P2 from Codex: tag name must be part of
    # the per-tag signature so an attribute moving between different
    # tag TYPES (``<img>`` vs ``<source>``) is still flagged even when
    # the raw attrs are permuted.
    source = '<img src="/a.png"><source src="/b.png">'
    decoded = '<img src="/b.png"><source src="/a.png">'
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "html_attr_tag_drift" in reason


def test_structural_position_allows_inline_tag_reorder():
    # Regression for cycle-6 P2 from Codex: legal cross-language
    # reorder of inline HTML fragments — two ``<a>`` tags swap order
    # to fit target grammar — must NOT be flagged.  Each attribute
    # stays with its original tag, so the per-tag signature multiset
    # is unchanged.
    source = 'Visit <a href="/a">A</a> and <a href="/b">B</a>.'
    decoded = (
        '<a href="/b">B</a>\uc640 <a href="/a">A</a>\ub97c '
        "\ubc29\ubb38\ud558\uc138\uc694."
    )
    assert check_structural_position(source, decoded) is None


def test_structural_position_ignores_heading_inside_fence():
    # Regression for cycle-17 P2 from Codex: ``_anchor_positions``
    # incremented heading ordinals for ``## fake`` lines sitting
    # inside fenced code blocks, which caused ``anchor_heading_drift``
    # reports when a code example changed between source and
    # translation while the real anchor stayed put.  Heading detection
    # now skips any line whose position is inside a code range.
    source = (
        "```md\n"
        "## fake heading\n"
        "```\n"
        "\n"
        "## Real {#real-id}"
    )
    decoded = (
        "```md\n"
        "## \ubcc0\ud658\ub41c \uc81c\ubaa9\n"  # translated code example
        "```\n"
        "\n"
        "## Real {#real-id}"
    )
    assert check_structural_position(source, decoded) is None


def test_structural_position_counts_setext_heading_as_heading():
    # Regression for cycle-5 P2 from Codex: setext headings (``Overview
    # {#overview}\n========``) were treated as prose by the positional
    # guard, letting a model move the anchor off the heading silently.
    # With setext detection, the anchor is pinned to the heading
    # ordinal and the move is flagged.
    source = "Overview {#overview}\n========"
    decoded = "\uac1c\uc694\n========\n\nText with {#overview} anchor."
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "anchor_heading_drift" in reason


def test_structural_position_setext_dash_heading_detected():
    # H2 setext underline (``----``) must count as a heading too.
    source = "Sub section {#sub}\n----"
    decoded = "\ud558\uc704 \uc139\uc158 {#sub}\n----"  # anchor stays
    assert check_structural_position(source, decoded) is None


def test_structural_position_tolerates_prose_reordering():
    # Anchors that were NOT on a heading line — span-level IAL inside
    # prose — may legitimately move with the prose across a language
    # reorder.  Only heading-anchors are pinned.
    source = "Some prose with {#span-id} anchor here."
    decoded = "\ub2e4\ub978 \uc5b8\uc5b4 prose {#span-id} with different order."
    assert check_structural_position(source, decoded) is None


def test_structural_position_empty_inputs_are_ok():
    # No anchors, no attrs → no drift → no reason.
    assert check_structural_position("", "") is None
    assert check_structural_position("plain text", "\ub2e4\ub978 \ud14d\uc2a4\ud2b8") is None


def test_processor_position_check_marks_entry_fuzzy_on_anchor_move():
    # End-to-end: _apply_validation must flag a block whose anchor
    # migrated off its heading line, even if the token count still
    # balances.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    source = "## Overview {#overview}"
    encoded, mapping = proc._encode_source(source)
    assert len(mapping) == 1
    # Simulate a response that echoes the token but in a different
    # structural position (off the heading line).
    moved_response = f"Something about overview {format_token(0)}"
    proc._tls.last_encoded_response = moved_response
    proc._tls.last_placeholder_map = mapping
    # Decoded translation has the anchor dangling at end of prose.
    decoded = "Something about overview {#overview}"

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(entry, decoded, inplace=False, pool=MagicMock())
    assert ok is False
    assert "fuzzy" in entry.flags
    assert "placeholder_position" in (entry.tcomment or "")


def test_user_same_name_override_without_opt_in_keeps_builtin_active():
    # Regression for cycle-19 P2 from Codex: registering an ``anchor``
    # pattern WITHOUT ``replace_builtin=True`` must NOT silently
    # disable the T-6 default.  The "always-on / no opt-out" contract
    # means every ``{#...}`` anchor is still tokenized even when the
    # user's regex covers only a subset.
    from mdpo_llm.processor import MarkdownProcessor

    narrow = re.compile(r"\{#pin-[\w-]+\}")
    user_reg = PlaceholderRegistry()
    user_reg.register("anchor", narrow)  # no replace_builtin flag

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    encoded, mapping = proc._encode_source(
        "## Alpha {#pin-alpha}\n\n## Beta {#free-beta}"
    )
    anchor_originals = sorted(
        p.original for p in mapping if p.pattern_name == "anchor"
    )
    # Both anchors protected — the default built-in layered in after
    # the user's narrow pattern and covered ``{#free-beta}``.
    assert anchor_originals == ["{#free-beta}", "{#pin-alpha}"]


def test_structural_position_rejects_block_ial_across_blank_line():
    # Regression for cycle-20 P2 from Codex: Kramdown's block IAL must
    # be on the IMMEDIATELY FOLLOWING line — intervening blank lines
    # break the association.  A standalone ``## Title\n\n{#id}`` must
    # NOT be pinned to that heading in the positional guard.
    source = "## Title\n\n{#free-id}"
    # The anchor legitimately moves with the prose across translation.
    decoded = "## \ubcc0\ud658\ub41c\n\nProse {#free-id} elsewhere"
    assert check_structural_position(source, decoded) is None


def test_html_attr_not_protected_in_nested_blockquoted_fence():
    # Regression for cycle-20 P2 from Codex: fences nested inside
    # multiple blockquote markers (``>> ~~~html``) must still be
    # recognised as code so HTML examples inside aren't tokenized.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(model="test-model", target_lang="ko")
    text = (
        ">> ~~~html\n"
        '>> <a href="/docs" class="bare">link</a>\n'
        ">> ~~~"
    )
    encoded, mapping = proc._encode_source(text)
    html_attr_hits = [p for p in mapping if p.pattern_name == "html_attr"]
    assert html_attr_hits == []


def test_structural_position_recognises_block_ial_heading_anchor():
    # Regression for cycle-19 P2 from Codex: Kramdown / Pandoc allow a
    # heading's IAL on the following line (``## Title\n{#id}``).  The
    # anchor is still heading-attached — ``_anchor_positions`` uses the
    # preceding-heading fallback so a relocation still fires
    # ``anchor_heading_drift``.
    source = "## Title\n{#pin-id}"
    decoded = "## Translated\n\nProse now mentions {#pin-id}"
    reason = check_structural_position(source, decoded)
    assert reason is not None
    assert "anchor_heading_drift" in reason


def test_user_placeholder_overrides_builtin_anchor_on_exact_match():
    # Regression for cycle-8 P2 from Codex: when a user re-registers
    # a pattern equivalent to a built-in (same regex) under a different
    # name or with a stricter predicate, the user's version must win on
    # equal-start / equal-length ties.  The ordering in
    # ``_build_effective_registry`` registers user patterns first so
    # PlaceholderRegistry's stable sort keeps them ahead of built-ins.
    from mdpo_llm.processor import MarkdownProcessor

    user_reg = PlaceholderRegistry()
    # Identical regex to the built-in anchor pattern, under a different
    # name.  On exact-match overlap the user's name must be the one
    # recorded in the mapping.
    user_reg.register("custom_anchor", ANCHOR_PATTERN)

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    encoded, mapping = proc._encode_source("## Overview {#overview}")
    assert len(mapping) == 1
    assert mapping.items[0].pattern_name == "custom_anchor"


def test_user_override_position_check_uses_user_regex():
    # Regression for cycle-10 + cycle-13 P2 from Codex: the position
    # check must run using the user's override regex — NOT the default
    # — so unrelated attribute shifts (``class=`` in this case) don't
    # trigger drift while the user's tokenized spans still get the
    # structural-safety guarantee.  ``replace_builtin=True`` is the
    # explicit opt-in to suppress the T-6 default (see cycle-19 P2).
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    # User override: ``html_attr`` protects ONLY ``href``.  Everything
    # else is out of scope for this configuration.
    href_only = re.compile(
        r'\bhref\s*=\s*(?:"[^"]*"|\'[^\']*\'|[^\s"\'=<>`]+)'
    )
    user_reg = PlaceholderRegistry()
    user_reg.register("html_attr", href_only, replace_builtin=True)

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    # Source has a href (protected) and a class (NOT protected under
    # the override) on the same tag.  Decoded shuffles the class to a
    # different tag — the default ``html_attr_tag_drift`` check would
    # fire on that.  With the override suppressing the default
    # position sub-check, validation must still pass.
    source = '<a href="/docs" class="primary">link</a><b class="big">B</b>'
    encoded, mapping = proc._encode_source(source)
    assert len(mapping) == 1
    # Simulate LLM output with href preserved but class attributes
    # relocated (legal under the override's intent).
    href_token = mapping.items[0].token
    encoded_response = f'<a {href_token} class="big">link</a><b class="primary">B</b>'
    decoded = f'<a href="/docs" class="big">link</a><b class="primary">B</b>'
    proc._tls.last_encoded_response = encoded_response
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(entry, decoded, inplace=False, pool=MagicMock())
    assert ok is True, entry.tcomment
    assert "fuzzy" not in entry.flags


def test_user_override_predicate_honoured_by_position_check():
    # Regression for cycle-14 P2 from Codex: when a user override
    # supplies a stricter predicate, ``check_structural_position`` must
    # respect it — otherwise spans the override deliberately skipped
    # are still scanned with the default rules and flagged, forcing
    # the entry fuzzy after a valid translation.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    # Override: only protect anchors whose id starts with "pin-".
    def only_pin_ids(text, start, end):
        return text[start + 2 : end - 1].startswith("pin-")

    user_reg = PlaceholderRegistry()
    user_reg.register(
        "anchor", ANCHOR_PATTERN, predicate=only_pin_ids, replace_builtin=True
    )

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    # Source has one pinned anchor and one un-pinned.  A valid
    # translation can legitimately relocate the un-pinned anchor
    # (the override chose not to protect it), and the position check
    # must not flag that move.
    source = "## Alpha {#pin-alpha}\n\n## Beta {#free-beta}"
    encoded, mapping = proc._encode_source(source)
    # Only the pinned one is encoded.
    assert [p.original for p in mapping] == ["{#pin-alpha}"]

    # Simulate output where the free anchor moved into prose, but the
    # pinned anchor stayed on its heading.
    token = mapping.items[0].token
    proc._tls.last_encoded_response = (
        f"## Alpha {token}\n\nProse now mentions {{#free-beta}}"
    )
    proc._tls.last_placeholder_map = mapping
    decoded = "## Alpha {#pin-alpha}\n\nProse now mentions {#free-beta}"

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(entry, decoded, inplace=False, pool=MagicMock())
    assert ok is True, entry.tcomment
    assert "fuzzy" not in entry.flags


def test_user_override_position_check_fires_on_actual_drift():
    # Regression for cycle-13 P2 from Codex: when a caller overrides
    # ``anchor`` with a narrower regex, the position check must keep
    # running against the USER'S match set — otherwise a response that
    # moves a protected ``{#ok-*}`` anchor off its heading passes
    # validation even though that span was still tokenized.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    narrower = re.compile(r"\{#ok-[\w-]+\}")
    user_reg = PlaceholderRegistry()
    user_reg.register("anchor", narrower, replace_builtin=True)

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    source = "## Overview {#ok-intro}"
    encoded, mapping = proc._encode_source(source)
    assert len(mapping) == 1

    # Simulate response that moves the anchor off the heading line.
    token = mapping.items[0].token
    proc._tls.last_encoded_response = f"Prose about {token}"
    proc._tls.last_placeholder_map = mapping

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(
        entry, "Prose about {#ok-intro}", inplace=False, pool=MagicMock()
    )
    assert ok is False
    assert "placeholder_position" in (entry.tcomment or "")


def test_multiple_same_name_overrides_all_guarded_by_position_check():
    # Regression for cycle-15 P2 from Codex: two user patterns under
    # the same built-in name must each keep their positional guarantee
    # — the processor runs ``check_structural_position`` once per
    # ``(regex, predicate)`` pair rather than dropping all but the
    # last.
    from unittest.mock import MagicMock

    import polib

    from mdpo_llm.processor import MarkdownProcessor

    # Two narrow anchor patterns, each covering a different id prefix.
    pin_re = re.compile(r"\{#pin-[\w-]+\}")
    keep_re = re.compile(r"\{#keep-[\w-]+\}")
    user_reg = PlaceholderRegistry()
    user_reg.register("anchor", pin_re, replace_builtin=True)
    user_reg.register("anchor", keep_re, replace_builtin=True)

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    source = "## Alpha {#pin-alpha}\n\n## Beta {#keep-beta}"
    encoded, mapping = proc._encode_source(source)
    # Both overrides fired — two tokens.
    originals = sorted(p.original for p in mapping)
    assert originals == ["{#keep-beta}", "{#pin-alpha}"]

    # Simulate a response that moved the FIRST-registered override's
    # anchor off its heading.  The position check must still flag
    # this, even though the second override is also in play.
    tok_map = {p.original: p.token for p in mapping}
    proc._tls.last_encoded_response = (
        f"## Alpha heading\n\nProse {tok_map['{#pin-alpha}']}\n\n"
        f"## Beta {tok_map['{#keep-beta}']}"
    )
    proc._tls.last_placeholder_map = mapping
    decoded = (
        "## Alpha heading\n\nProse {#pin-alpha}\n\n## Beta {#keep-beta}"
    )

    entry = polib.POEntry(msgid=source, msgstr="")
    ok = proc._apply_validation(entry, decoded, inplace=False, pool=MagicMock())
    assert ok is False
    assert "placeholder_position" in (entry.tcomment or "")


def test_user_same_name_override_replaces_builtin_entirely():
    # Regression for cycle-9 P2 from Codex: registering a user pattern
    # under the built-in name (``anchor``) with a stricter predicate
    # must actually narrow the match set — the built-in default must
    # NOT re-add the rejected spans.  Before this fix the built-in
    # was still registered alongside the user pattern, so any span the
    # user's predicate rejected was immediately picked up by the
    # defaults.
    from mdpo_llm.processor import MarkdownProcessor

    # Stricter override: accept only anchors whose id starts with "ok-".
    def only_ok_ids(text, start, end):
        return text[start + 2 : end - 1].startswith("ok-")

    user_reg = PlaceholderRegistry()
    user_reg.register(
        "anchor", ANCHOR_PATTERN, predicate=only_ok_ids, replace_builtin=True
    )

    proc = MarkdownProcessor(
        model="test-model", target_lang="ko", placeholders=user_reg
    )
    encoded, mapping = proc._encode_source(
        "## Intro {#ok-intro}\n\n## Next {#skip-next}"
    )
    anchors = [p for p in mapping if p.pattern_name == "anchor"]
    # Only the ``{#ok-intro}`` anchor is protected; the built-in did
    # NOT fall back to protect ``{#skip-next}`` despite the user's
    # predicate rejecting it.
    assert [p.original for p in anchors] == ["{#ok-intro}"]


def test_builtin_layered_with_user_and_glossary_patterns():
    # User-supplied patterns still fire on top of the built-ins, and so
    # do glossary placeholder patterns.  Verifies the three-layer
    # composition the T-6 refactor introduced.
    from mdpo_llm.processor import MarkdownProcessor

    user_reg = PlaceholderRegistry()
    user_reg.register("url", r"https?://\S+")

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        placeholders=user_reg,
        glossary={"GitHub": None},
        glossary_mode="placeholder",
    )
    text = 'See https://example.com on GitHub via <a class="x">link</a> {#link}'
    encoded, mapping = proc._encode_source(text)

    names = sorted({p.pattern_name for p in mapping})
    # All four layers landed at least one match.
    assert "anchor" in names  # built-in
    assert "html_attr" in names  # built-in
    assert "url" in names  # user
    assert any(n.startswith("glossary:") for n in names)

    # Decode restores every protected span so the full document is a
    # byte-for-byte round trip (do-not-translate glossary term included).
    assert PlaceholderRegistry.decode(encoded, mapping) == text
