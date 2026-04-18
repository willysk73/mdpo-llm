"""Unit tests for the placeholder substitution framework (T-4)."""

from __future__ import annotations

import re

import pytest

from mdpo_llm.placeholder import (
    Placeholder,
    PlaceholderMap,
    PlaceholderRegistry,
    TOKEN_RE,
    check_round_trip,
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


def test_glossary_mode_defaults_to_instruction():
    """Default mode preserves v0.4 back-compat: glossary block in the system
    prompt, terms untouched in the user message."""
    from unittest.mock import MagicMock, patch

    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary={"GitHub": None, "API": "API"},
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
    # Placeholder mode without a glossary is a no-op — no effective
    # registry, no substitution.
    from mdpo_llm.processor import MarkdownProcessor

    proc = MarkdownProcessor(
        model="test-model",
        target_lang="ko",
        glossary_mode="placeholder",
    )
    encoded, mapping = proc._encode_source("unchanged text")
    assert encoded == "unchanged text"
    assert mapping is None


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
