"""
Document reconstruction from translated PO entries.
"""

from typing import Any, Dict, List

import polib


class DocumentReconstructor:
    """Reconstructs translated Markdown documents from PO files."""

    def __init__(self, skip_types: List[str] = []):
        """Initialize document reconstructor."""
        self.skip_types = skip_types

    def rebuild_markdown(
        self,
        source_lines: List[str],
        blocks: List[Dict[str, Any]],
        po_file: polib.POFile,
        context_id_func,
    ) -> str:
        """
        Rebuild processed Markdown from source lines, blocks, and PO file.

        Args:
            source_lines: Original markdown lines (with newlines)
            blocks: Parsed blocks from BlockParser
            po_file: PO file with processed entries
            context_id_func: Function to generate context IDs for blocks

        Returns:
            Processed markdown as string
        """
        output = []
        position = 0

        for block in blocks:
            # Copy any untouched gap between blocks
            if position < block["start"]:
                output.extend(source_lines[position : block["start"]])

            context_id = context_id_func(block)
            entry = next(
                (e for e in po_file if e.msgctxt == context_id and not e.obsolete), None
            )

            if block["type"] in self.skip_types:
                # Copy original content as-is
                output.extend(source_lines[block["start"] : block["end"]])
            else:
                # Handle processable blocks
                if entry and entry.msgstr:
                    # Use processing
                    translated_lines = [
                        line + "\n" for line in entry.msgstr.split("\n")
                    ]
                    output.extend(translated_lines)
                else:
                    # Fallback: copy original content
                    output.extend(source_lines[block["start"] : block["end"]])

            position = block["end"]

        # Copy any remaining content after the last block
        if position < len(source_lines):
            output.extend(source_lines[position:])

        return "".join(output)

    def get_process_coverage(
        self, blocks: List[Dict[str, Any]], po_file: polib.POFile, context_id_func
    ) -> Dict[str, Any]:
        """
        Get translation coverage statistics.

        Args:
            blocks: Parsed blocks
            po_file: PO file with translations
            context_id_func: Function to generate context IDs

        Returns:
            Dictionary with coverage statistics
        """
        stats = {
            "total_blocks": len(blocks),
            "translatable_blocks": 0,
            "translated_blocks": 0,
            "fuzzy_blocks": 0,
            "untranslated_blocks": 0,
            "coverage_percentage": 0.0,
            "by_type": {},
        }

        type_stats = {}

        for block in blocks:
            block_type = block["type"]
            if block_type not in type_stats:
                type_stats[block_type] = {
                    "total": 0,
                    "translatable": 0,
                    "translated": 0,
                    "fuzzy": 0,
                }

            type_stats[block_type]["total"] += 1

            # Skip non-translatable blocks
            if block_type in self.skip_types:
                continue

            type_stats[block_type]["translatable"] += 1
            stats["translatable_blocks"] += 1

            context_id = context_id_func(block)
            entry = next(
                (e for e in po_file if e.msgctxt == context_id and not e.obsolete), None
            )

            if entry and entry.msgstr:
                if "fuzzy" in entry.flags:
                    stats["fuzzy_blocks"] += 1
                    type_stats[block_type]["fuzzy"] += 1
                else:
                    stats["translated_blocks"] += 1
                    type_stats[block_type]["translated"] += 1
            else:
                stats["untranslated_blocks"] += 1

        # Calculate coverage percentage
        if stats["translatable_blocks"] > 0:
            stats["coverage_percentage"] = (
                stats["translated_blocks"] / stats["translatable_blocks"]
            ) * 100

        stats["by_type"] = type_stats

        return stats

    def export_translation_report(
        self,
        source_file: str,
        blocks: List[Dict[str, Any]],
        po_file: polib.POFile,
        context_id_func,
    ) -> str:
        """
        Export detailed translation report.

        Args:
            source_file: Path to source file
            blocks: Parsed blocks
            po_file: PO file with translations
            context_id_func: Function to generate context IDs

        Returns:
            Detailed report as string
        """
        coverage = self.get_process_coverage(blocks, po_file, context_id_func)

        report_lines = []
        report_lines.append("# Translation Report\n")
        report_lines.append(f"**Source File:** {source_file}\n")
        report_lines.append(
            f"**Generated:** {po_file.metadata.get('POT-Creation-Date', 'Unknown')}\n\n"
        )

        # Summary statistics
        report_lines.append("## Summary\n")
        report_lines.append(f"- **Total Blocks:** {coverage['total_blocks']}\n")
        report_lines.append(
            f"- **Translatable Blocks:** {coverage['translatable_blocks']}\n"
        )
        report_lines.append(f"- **Translated:** {coverage['translated_blocks']}\n")
        report_lines.append(f"- **Fuzzy:** {coverage['fuzzy_blocks']}\n")
        report_lines.append(f"- **Untranslated:** {coverage['untranslated_blocks']}\n")
        report_lines.append(
            f"- **Coverage:** {coverage['coverage_percentage']:.1f}%\n\n"
        )

        # By block type
        report_lines.append("## By Block Type\n")
        for block_type, type_stats in coverage["by_type"].items():
            if type_stats["translatable"] > 0:
                coverage_pct = (
                    type_stats["translated"] / type_stats["translatable"]
                ) * 100
                report_lines.append(
                    f"- **{block_type.title()}:** {type_stats['translated']}/{type_stats['translatable']} ({coverage_pct:.1f}%)\n"
                )
            else:
                report_lines.append(
                    f"- **{block_type.title()}:** {type_stats['total']} (non-translatable)\n"
                )

        return "".join(report_lines)
