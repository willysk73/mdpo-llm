from contracts import LLMInterface


class MyLLM(LLMInterface):
    def process(self, source_text: str) -> str:
        """
        Example implementation of the process method.
        This method should contain the logic to process the source_text.
        e.g., calling an external API or performing translation or refinement.

        Args:
            source_text (str): The text to be processed.
        Returns:
            str: The processed text.
        """
        # processed_text = self._call_external_service(source_text)
        processed_text = f"Processed: {source_text}"
        return processed_text


def main():
    print("Hello from mdpo-llm!")


if __name__ == "__main__":
    main()
