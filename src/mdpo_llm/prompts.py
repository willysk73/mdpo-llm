class Prompts:
    TRANSLATE_SYSTEM_TEMPLATE = (
        "You are an expert technical translator. "
        "Translate the following content into **{lang}**.\n"
        "Output only the translated result. "
        "Do not include any explanations or comments — only the translation.\n\n"
        "{instruction}\n"
    )

    TRANSLATE_INSTRUCTION = (
        "Translate the following technical Markdown document into the target language.\n\n"
        "Rules:\n"
        "1. Preserve all Markdown formatting exactly (headings, lists, tables, links, bold, italic, etc.).\n"
        "2. Translate human-readable prose: headings, paragraphs, list items, table cells, blockquotes.\n"
        "3. In code blocks: keep all code as-is. Only translate comments and user-facing string literals.\n"
        "4. Keep inline code unchanged unless it contains human-readable prose (e.g., UI labels).\n"
        "5. Keep URLs, file paths, and variable/function names unchanged.\n"
        "6. Preserve interpolation tokens and placeholders as-is (e.g., `{{name}}`, `%s`, `${{var}}`).\n"
        "7. Widely-adopted technical terms (e.g., API, SDK, GPU) may remain in English "
        "if that is conventional in the target language.\n"
    )
