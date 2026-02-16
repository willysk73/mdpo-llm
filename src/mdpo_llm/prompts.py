class Prompts:
    TRANSLATE_SYSTEM_TEMPLATE = (
        "You are an expert technical translator. Translate the following content into **{lang}**.\n"
        "Following the instruction, translate the text into **{lang}**.\n"
        "Output only the translated result. Do not include any explanations, change logs, or additional comments—only the translation.\n"
        "Instruction:\n{instruction}\n"
    )
    VALIDATE_HUMAN_TEMPLATE = (
        "Assess the OUTPUT against the SOURCE document.\n\n"
        "Target language: **{lang}**\n\n"
        "Criteria for a **'yes'** score (ALL must be satisfied):\n"
        "{criteria}\n\n"
        "SOURCE:\n{source}\n\n"
        "OUTPUT:\n{processed}\n\n"
        'Respond with JSON → "reason":"…","binary_score":"yes|no"'
    )
    TRANSLATE_INSTRUCTION = (
        "Translate the following technical/developer Markdown into the target language. "
        "Preserve every Markdown construct exactly as-is (headings, lists, tables, fenced code, etc.).\n"
        "• **Fenced code blocks (` ``` `):** Keep code structure and identifiers unchanged. "
        "  Translate **comment text**, and also **string literals that are clearly user-facing placeholders or messages**. "
        "  Do **NOT** translate identifiers, keywords, function/variable names, file names, or code-like constants.\n"
        "• **Inline code (` `):** Translate the content inside backticks **ONLY IF** it is human-readable prose (e.g., UI labels like `` `Save` ``, status like `` `Completed` ``). "
        "  Leave it unchanged if it is a code identifier (e.g., variable `userId`, function `calculate()`, filename `config.json`).\n"
        "• **URLs:** You may translate human-readable path segments or query-string values. Keep the rest of the URL intact.\n"
        "• **Unified placeholder/string rule (applies to JSON/YAML/data AND code):**\n"
        '   - **Translate** string values that are clearly user-facing **placeholders or messages**, such as "Enter your name", "Please select...", '
        '     "Your token here", "e.g., user@example.com", "Error: invalid token".\n'
        "   - **Preserve** keys, identifiers, and interpolation tokens (e.g., `{{name}}`, `${{var}}`, `%s`, `%1$s`, `{{0}}`, `:id`, `<id>`). "
        '     Translate only the surrounding human text (e.g., "Enter {{name}}" → translate "Enter" but keep `{{name}}`).\n'
        '   - **Do not translate** code-ish identifiers or config-like values (e.g., "user_id", "config_path", "AuthError"), even if they appear as strings.\n'
        "• **Translatable segments:** Human-readable prose in headings/paragraphs/list items/table cells/alt text, comment text inside code blocks, "
        "**and** user-facing/placeholder strings in data files **and in code** (per the unified rule above).\n"
        "• **Table headers:** Column headers in Markdown tables are user-facing labels and must be translated into the target language, "
        "  unless they are established technical terms (e.g., 'API', 'GPU').\n\n"
        "If the previous version was rejected, correct every issue mentioned in the feedback.\n\n"
        "Reason for previous rejection (if any): {reason}\n"
    )

    VALIDATE_SYSTEM = (
        "You are an expert Markdown and translation validator. "
        "Return **only** a JSON object with two keys: "
        "'binary_score' (\"yes\" or \"no\") and 'reason' (briefly justify your decision)."
    )

    VALIDATE_CRITERIA = (
        "1. **Markdown integrity** – All Markdown structures (headings, lists, tables, fenced code, links) are preserved.\n"
        "2. **Code integrity rule** –\n"
        "   - **Fenced code blocks (` ``` `):** Code structure and identifiers are unchanged. "
        "     Only **comment text** and **clearly user-facing string literals** may be translated; all other code remains identical.\n"
        "   - **Inline code (` `):** Translate only when it contains human-readable prose (UI labels/status). "
        "     Leave code identifiers (variables, functions, filenames) unchanged.\n"
        "3. **URL rule** – URL structure preserved; human-readable path/query values may be translated; scheme/host and keys remain intact.\n"
        "4. **Translatability gating** – First determine whether the SOURCE contains any **human-readable prose** "
        "(UI labels, messages, descriptions, alt text, placeholders, **comment text in code**, or **user-facing string literals in code/data**). "
        "If **none** exists, an identical TRANSLATION is acceptable and should **PASS**.\n"
        "5. **Target-language requirement** – If ≥1 translatable segment exists, those segments must be correctly rendered in **{lang}** "
        "(segments should not remain identical to SOURCE unless the SOURCE is already in {lang}). "
        "Non-translatable content must remain identical.\n"
        "6. **Complete & accurate translation** – All human-readable content is fully and correctly rendered in **{lang}**.\n"
        "7. **Terminology leniency** – Widely-adopted technical terms (e.g., 'Cookie', 'API', 'Server', 'GPU') "
        "may remain in English **if they are conventionally used that way in {lang}**.\n"
        "8. **Markdown table headers** – Column headers in Markdown tables are user-facing labels and must be translated into **{lang}**, "
        "unless they are established technical terms covered by rule 7.\n"
        "9. **Unified placeholder/string rule (validation):**\n"
        "   - **Translate** user-facing placeholders/messages whether they appear in data files (JSON/YAML/etc.) or **inside code** as string literals.\n"
        "   - **Preserve** keys/identifiers and interpolation tokens exactly (`{{name}}`, `${{var}}`, `%s`, `%1$s`, `{{0}}`, `:id`, `<id>`); "
        "     only the surrounding human text may be translated.\n"
        '   - **Do not translate** non-user-facing identifiers/config-like strings (e.g., "user_id", "config_path").\n'
    )

