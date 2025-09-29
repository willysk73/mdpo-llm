import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from contracts import LLMInterface
from contracts.language import LanguageCode
from po import MarkdownProcessor

logger = logging.getLogger(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
mapping_path = os.path.join(script_dir, "filename_map.json")

with open(mapping_path, "r", encoding="utf-8") as f:
    MAPPING: List[Dict] = json.load(f)


def find_translation_filename(
    filename: str, target_lang: LanguageCode
) -> Optional[str]:
    target_locale = target_lang.name.lower()
    for e in MAPPING:
        if e.get("source") == filename:
            return (e.get("translations") or {}).get(target_locale)

    return None


def _process(
    llm_interface: LLMInterface,
    lang: LanguageCode,
    input_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    processor = MarkdownProcessor(llm_interface=llm_interface)
    filename = input_path.name
    inplace = False
    if lang == LanguageCode.KO:
        inplace = True
    else:
        filename = find_translation_filename(filename, lang)
        if not filename:
            raise ValueError(
                f"No translation mapping found for {input_path.name} to {lang}"
            )
    output_path = output_dir / filename
    po_path = output_path.with_suffix(".po")

    return processor.process_document(input_path, output_path, po_path, inplace)


def args_parser():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--to",
        type=str,
        help=f"Language code for the output ({LanguageCode._member_names_})",
        default="ko",
        required=False,
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Enable fast mode",
    )
    parser.add_argument(
        "--env",
        action="store_true",
        help="Enable env mode",
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Path to the original Markdown file",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to the refined Markdown file",
        default="",
        required=False,
    )
    return parser.parse_args()


def validate_args(args) -> bool:
    """Validate CLI arguments."""

    if not args.input or not args.output_dir:
        print("Error: Source and output directory required")
        return False

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Source file not found: {input_path}")
        return False

    if not input_path.suffix == ".md":
        print(f"Error: Source file must be a Markdown file: {input_path}")
        return False

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        print(f"Warning: Output directory does not exist. Creating: {output_dir}")
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Error: Failed to create output directory: {str(e)}")
            return False

    lang = args.to.upper()
    if lang not in LanguageCode._member_names_:
        print(f"Error: Unsupported language code: {lang}")
        return False

    return True


def process(llm_interface: LLMInterface):
    args = args_parser()
    validate_args(args)

    language = LanguageCode[args.to.upper()]

    logger.info(f"Processing {args.input} to {args.output_dir} as {language}")

    results = _process(
        llm_interface,
        lang=language,
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
    )
    logger.info("Process completed: %s", results)
