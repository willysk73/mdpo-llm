class Prompts:
    TRANSLATE_SYSTEM_TEMPLATE = (
        "You are an expert technical translator. Translate the following content into **{lang}**.\n"
        "Following the instruction, translate the text into **{lang}**.\n"
        "Output only the translated result. Do not include any explanations, change logs, or additional comments—only the translation.\n"
        "Instruction:\n{system_prompt}\n\n"
        "Source text:\n{source}"
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
    REFINE_SYSTEM_TEMPLATE = (
        "You are an expert technical **editor**. Refine the following text written in **{lang}**.\n\n"
        "Your tasks:\n"
        "1. Correct spelling, grammar, punctuation, and word choice.\n"
        "2. Improve clarity, flow, and overall readability while **preserving original meaning**.\n"
        "3. Keep all Markdown structures (headings, code blocks, lists, links, tables, etc.) exactly as they appear.\n"
        "4. Do **not** translate, summarize, or add new information unless explicitly instructed.\n"
        "Output only the refined result. Do not include any explanations, change logs, or additional comments—only the refined text.\n\n"
        "Instruction: {system_prompt}"
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

    REFINE_INSTRUCTION = (
        "You are an expert technical writer. Refine the following Markdown document **without altering meaning**. "
        "Focus only on improving the natural language text (human-readable parts), including comments inside code blocks.\n\n"
        "**STYLE CONSISTENCY WITH PREVIOUS CHUNK (if provided):**\n"
        "• A previous refined chunk may be provided as a style anchor.\n"
        "• Match its tone, voice (active/passive), person (e.g., second person vs. neutral), sentence length, and formality.\n"
        "• Mirror punctuation habits in lists and sentences (e.g., whether list items end with periods), and capitalization style for terms.\n"
        "• Prefer the same terminology/wording for the same concepts to keep consistency across chunks.\n"
        "• Never restructure Markdown to force consistency; adjust wording only.\n\n"
        "**CORE RULES (NEVER VIOLATE):**\n"
        "• Preserve ALL Markdown structure exactly (headings, lists, tables, links, bold, italics, etc.)\n"
        "• Inside code blocks: You may revise comments for clarity/grammar, but DO NOT change code, identifiers, order, or formatting\n"
        "• URLs: Keep structure intact; change only to correct obvious typos\n"
        "• Meaning must remain 100% identical - no additions, removals, or changes\n\n"
        "**LANGUAGE IMPROVEMENTS:**\n"
        "• Remove redundant words and phrases\n"
        "• Fix grammatical errors and typos\n"
        "• Improve sentence flow and clarity\n"
        "• Use active voice where appropriate (unless the style anchor consistently uses another voice)\n"
        "• Ensure consistent terminology throughout, aligning with the style anchor when present\n"
        "• Maintain a professional, concise tone suitable for technical documentation\n\n"
        "Work methodically through the entire document. Miss nothing.\n"
        "Reason for previous rejection (if any): {reason}\n"
        "Previous refined chunk (optional, style anchor): {previous_chunk}\n"
    )

    REFINEMENT_VALIDATE_SYSTEM = (
        "You are a strict Markdown quality validator. Evaluate the refinement against ALL criteria. "
        "Be thorough and critical - if ANY criterion fails, score 'no'. "
        "Respond **only** with a JSON object containing: "
        "'binary_score' (\"yes\" or \"no\") and 'reason' (brief justification, especially if 'no')."
    )

    REFINEMENT_VALIDATE_CRITERIA = (
        "**VALIDATION CHECKLIST - ALL must pass:**\n\n"
        "1. **Structural Integrity**\n"
        "   - All headings, lists, tables, links, bold/italic text preserved exactly\n"
        "   - Markdown syntax unchanged in structure\n"
        "   - No missing or added content sections\n\n"
        "2. **Code Block Compliance**\n"
        "   - Only comments within code blocks may be edited\n"
        "   - Code itself (logic, identifiers, order, formatting) remains unchanged\n"
        "   - Fenced block delimiters and language identifiers remain unchanged\n\n"
        "3. **Language Quality**\n"
        "   - Grammar and spelling correct\n"
        "   - Concise, professional tone\n"
        "   - Consistent terminology\n"
        "   - Clear, readable sentences\n\n"
        "4. **Semantic Preservation**\n"
        "   - Meaning identical to original\n"
        "   - No information added, removed, or altered\n"
        "   - Technical accuracy maintained\n\n"
        "5. **URL Integrity**\n"
        "   - URLs preserved; only correct obvious typos if necessary\n\n"
        "6. **Style Consistency with Previous Chunk (if provided)**\n"
        "   - Tone, voice, and formality align with the provided style anchor\n"
        "   - Terminology matches prior choices for the same concepts\n"
        "   - Punctuation habits (e.g., periods in list items) mirror the style anchor\n\n"
        "**FAIL immediately if:**\n"
        "- Any change to code (non-comment) inside code blocks\n"
        "- Structural changes to Markdown elements\n"
        "- Meaning alterations of any kind\n"
        "- Addition or removal of content beyond comment wording improvements\n\n"
        "Previous refined chunk (optional, style anchor): {previous_chunk}\n"
    )
