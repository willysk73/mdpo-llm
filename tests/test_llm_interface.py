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
