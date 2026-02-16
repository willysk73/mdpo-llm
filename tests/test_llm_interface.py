"""Tests for LLM interface abstraction."""

import pytest

from mdpo_llm.llm_interface import LLMInterface, MockLLMInterface


class TestLLMInterfaceAbstract:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            LLMInterface()

    def test_subclass_must_implement_process(self):
        class IncompleteLLM(LLMInterface):
            pass

        with pytest.raises(TypeError):
            IncompleteLLM()

    def test_subclass_with_process_works(self):
        class ConcreteLLM(LLMInterface):
            def process(self, source_text: str) -> str:
                return source_text.upper()

        llm = ConcreteLLM()
        assert llm.process("hello") == "HELLO"


class TestMockLLMInterface:
    def test_prefix_applied(self):
        mock = MockLLMInterface()
        result = mock.process("Hello world")
        assert result == "[MOCK PROCESSING] Hello world"

    def test_empty_string(self):
        mock = MockLLMInterface()
        result = mock.process("")
        assert result == "[MOCK PROCESSING] "

    def test_multiline(self):
        mock = MockLLMInterface()
        text = "Line 1\nLine 2\nLine 3"
        result = mock.process(text)
        assert result == f"[MOCK PROCESSING] {text}"

    def test_unicode(self):
        mock = MockLLMInterface()
        text = "한국어 텍스트 你好世界"
        result = mock.process(text)
        assert result == f"[MOCK PROCESSING] {text}"

    def test_is_instance_of_llm_interface(self):
        mock = MockLLMInterface()
        assert isinstance(mock, LLMInterface)

    def test_without_reference_pairs(self):
        mock = MockLLMInterface()
        result = mock.process("Hello")
        assert result == "[MOCK PROCESSING] Hello"

    def test_with_reference_pairs(self):
        mock = MockLLMInterface()
        pairs = [("src1", "tgt1"), ("src2", "tgt2")]
        result = mock.process("Hello", reference_pairs=pairs)
        assert result == "[MOCK PROCESSING ref=2] Hello"

    def test_with_empty_reference_pairs(self):
        mock = MockLLMInterface()
        result = mock.process("Hello", reference_pairs=[])
        # Empty list is falsy, so should return the no-ref format
        assert result == "[MOCK PROCESSING] Hello"

    def test_with_none_reference_pairs(self):
        mock = MockLLMInterface()
        result = mock.process("Hello", reference_pairs=None)
        assert result == "[MOCK PROCESSING] Hello"


class TestMockLLMTargetLang:
    def test_target_lang_only(self):
        mock = MockLLMInterface()
        result = mock.process("Hello", target_lang="ko")
        assert result == "[MOCK PROCESSING lang=ko] Hello"

    def test_target_lang_with_reference_pairs(self):
        mock = MockLLMInterface()
        pairs = [("a", "b")]
        result = mock.process("Hello", reference_pairs=pairs, target_lang="ja")
        assert result == "[MOCK PROCESSING lang=ja ref=1] Hello"

    def test_target_lang_none_omitted(self):
        mock = MockLLMInterface()
        result = mock.process("Hello", target_lang=None)
        assert result == "[MOCK PROCESSING] Hello"

    def test_target_lang_empty_string_omitted(self):
        mock = MockLLMInterface()
        result = mock.process("Hello", target_lang="")
        # Empty string is falsy
        assert result == "[MOCK PROCESSING] Hello"


class TestOldStyleSubclass:
    """Backward compatibility: subclasses without reference_pairs still work."""

    def test_old_style_subclass(self):
        class OldLLM(LLMInterface):
            def process(self, source_text: str) -> str:
                return source_text.upper()

        llm = OldLLM()
        assert llm.process("hello") == "HELLO"

    def test_old_style_with_reference_pairs_only(self):
        """Subclass accepting reference_pairs but not target_lang still works."""

        class RefOnlyLLM(LLMInterface):
            def process(self, source_text: str, reference_pairs=None) -> str:
                if reference_pairs:
                    return f"[REF={len(reference_pairs)}] {source_text}"
                return source_text

        llm = RefOnlyLLM()
        assert llm.process("hello", reference_pairs=[("a", "b")]) == "[REF=1] hello"
