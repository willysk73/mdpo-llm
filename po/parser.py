"""
Markdown block parser for translation workflow.
"""

import re
from typing import Any, Dict, List


class BlockParser:
    """Parses Markdown documents into semantic blocks for translation."""

    FENCE_RE = re.compile(r"^(```|~~~)")  # code fences
    YAML_DELIMITER = "---"

    def __init__(self):
        self.slug_counters = {}  # Track slug usage by level: {level: {slug: count}}

    def slugify(self, s: str) -> str:
        """Convert text to a URL-friendly slug."""
        s = re.sub(r"[^\w\s-]", "", s.lower())
        s = re.sub(r"\s+", "-", s).strip("-")
        return s or "section"

    def context_id(self, block: Dict[str, Any]) -> str:
        """Generate unique context ID for a block."""
        return f"{'/'.join(block['path'])}::{block['type']}:{block['idx_in_section']}"

    def segment_markdown(self, lines: List[str]) -> List[Dict[str, Any]]:
        """
        Parse markdown lines into semantic blocks.

        Returns list of blocks:
        [{ 'type': 'heading'|'para'|'ulist'|'olist'|'quote'|'table'|'code'|'hr',
           'text': '...', 'start': i, 'end': j, 'path': ['h1','h2',...], 'idx_in_section': k }]
        """
        blocks = []
        path = []  # heading slug stack
        self.slug_counters = {}  # Reset for each document
        i = 0

        while i < len(lines):
            line = lines[i]

            # code fences (copy as raw; don't translate)
            if self.FENCE_RE.match(line.strip()):
                i = self._parse_code_block(lines, i, blocks, path)
                continue

            # headings
            heading_match = re.match(r"^(#{1,6})\s+(.*)", line)
            if heading_match:
                i = self._parse_heading(lines, i, blocks, path, heading_match)
                continue

            # horizontal rules
            if re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", line):
                blocks.append(
                    {
                        "type": "hr",
                        "text": line.rstrip("\n"),
                        "start": i,
                        "end": i + 1,
                        "path": path.copy(),
                    }
                )
                i += 1
                continue

            # blockquote
            if line.lstrip().startswith(">"):
                i = self._parse_blockquote(lines, i, blocks, path)
                continue

            # list (bulleted/numbered)
            if re.match(r"^\s*([-*+]|\d+\.)\s+", line):
                i = self._parse_list(lines, i, blocks, path)
                continue

            # table (very simple heuristic: pipes on multiple lines)
            if "|" in line and re.match(r"^\s*\|", line):
                i = self._parse_table(lines, i, blocks, path)
                continue

            # paragraph or blank
            if line.strip() == "":
                i += 1
            else:
                i = self._parse_paragraph(lines, i, blocks, path)

        # add per-section indices
        self._add_section_indices(blocks)
        return blocks

    def _parse_code_block(
        self, lines: List[str], start: int, blocks: List[Dict], path: List[str]
    ) -> int:
        """Parse a fenced code block."""
        line = lines[start]
        fence = line.strip()[:3]
        i = start + 1
        while i < len(lines) and not lines[i].strip().startswith(fence):
            i += 1
        i = min(i + 1, len(lines))
        blocks.append(
            {
                "type": "code",
                "text": "\n".join(lines[start:i]),
                "start": start,
                "end": i,
                "path": path.copy(),
            }
        )
        return i

    def _parse_heading(
        self, lines: List[str], start: int, blocks: List[Dict], path: List[str], match
    ) -> int:
        """Parse a heading and update the path."""
        level = len(match.group(1))
        title = match.group(2).strip()
        base_slug = self.slugify(title)

        # Initialize level counter if not exists
        if level not in self.slug_counters:
            self.slug_counters[level] = {}

        # Make slug unique at this level
        if base_slug not in self.slug_counters[level]:
            self.slug_counters[level][base_slug] = 0
            unique_slug = base_slug
        else:
            self.slug_counters[level][base_slug] += 1
            unique_slug = f"{base_slug}-{self.slug_counters[level][base_slug]}"

        # Clear deeper level counters (when going from h3 back to h1, clear h2, h3, etc)
        levels_to_clear = [l for l in self.slug_counters.keys() if l > level]
        for l in levels_to_clear:
            del self.slug_counters[l]

        # Update path
        path[:] = path[: level - 1] + [unique_slug]
        blocks.append(
            {
                "type": "heading",
                "text": lines[start].rstrip("\n"),
                "start": start,
                "end": start + 1,
                "path": path.copy(),
            }
        )
        return start + 1

    def _parse_blockquote(
        self, lines: List[str], start: int, blocks: List[Dict], path: List[str]
    ) -> int:
        """Parse a blockquote."""
        chunk = [lines[start].rstrip("\n")]
        i = start + 1
        while i < len(lines) and lines[i].lstrip().startswith(">"):
            chunk.append(lines[i].rstrip("\n"))
            i += 1
        blocks.append(
            {
                "type": "quote",
                "text": "\n".join(chunk),
                "start": start,
                "end": i,
                "path": path.copy(),
            }
        )
        return i

    def _parse_list(
        self, lines: List[str], start: int, blocks: List[Dict], path: List[str]
    ) -> int:
        """Parse a list (bulleted or numbered) with proper line break handling."""
        chunk = [lines[start].rstrip("\n")]
        i = start + 1

        # Determine list type from first line
        first_line = lines[start]
        is_ordered = bool(re.match(r"^\s*\d+\.", first_line))
        list_type = "olist" if is_ordered else "ulist"

        # Get indentation level of first item
        first_match = re.match(r"^(\s*)([-*+]|\d+\.)", first_line)
        base_indent = len(first_match.group(1)) if first_match else 0

        while i < len(lines):
            line = lines[i]

            # Check if this is a new list item
            list_match = re.match(r"^(\s*)([-*+]|\d+\.)\s+", line)
            if list_match:
                indent_level = len(list_match.group(1))
                current_is_ordered = bool(re.match(r"^\s*\d+\.", line))

                # Stop if list type changes at same indentation level
                if indent_level == base_indent and current_is_ordered != is_ordered:
                    break

                # Stop if indentation is less than base (parent level)
                if indent_level < base_indent:
                    break

                chunk.append(line.rstrip("\n"))
                i += 1
                continue

            # Handle empty lines and continuation
            if line.strip() == "":
                # Look ahead to see if list continues
                next_i = i + 1
                while next_i < len(lines) and lines[next_i].strip() == "":
                    next_i += 1

                if next_i < len(lines) and re.match(
                    r"^\s*([-*+]|\d+\.)\s+", lines[next_i]
                ):
                    # Check if next list item is same type
                    next_is_ordered = bool(re.match(r"^\s*\d+\.", lines[next_i]))
                    next_match = re.match(r"^(\s*)([-*+]|\d+\.)", lines[next_i])
                    next_indent = len(next_match.group(1)) if next_match else 0

                    if next_indent == base_indent and next_is_ordered != is_ordered:
                        # Different list type at same level - stop here
                        break

                    chunk.append(line.rstrip("\n"))
                    i += 1
                    continue
                else:
                    # No more list items following
                    break

            # Handle continuation lines (indented content or unindented continuation)
            if line.strip() and not self._is_other_block_start(line):
                # Check if it's properly indented continuation
                if len(line) > base_indent and line.startswith(" " * (base_indent + 2)):
                    chunk.append(line.rstrip("\n"))
                    i += 1
                    continue
                # Check if it's an unindented continuation (common in Korean/Asian text)
                elif not re.match(
                    r"^\s*([-*+]|\d+\.)\s+", line
                ) and not line.startswith("#"):
                    # This might be a continuation line - include it
                    chunk.append(line.rstrip("\n"))
                    i += 1
                    continue

            # Line doesn't belong to this list
            break

        blocks.append(
            {
                "type": list_type,
                "text": "\n".join(chunk),
                "start": start,
                "end": i,
                "path": path.copy(),
            }
        )
        return i

    def _is_other_block_start(self, line: str) -> bool:
        """Check if line starts a different block type."""
        return any(
            [
                self.FENCE_RE.match(line.strip()),
                re.match(r"^(#{1,6})\s+", line),
                line.lstrip().startswith(">"),
                re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", line),
                ("|" in line and re.match(r"^\s*\|", line)),
            ]
        )

    def _parse_table(
        self, lines: List[str], start: int, blocks: List[Dict], path: List[str]
    ) -> int:
        """Parse a table."""
        chunk = [lines[start].rstrip("\n")]
        i = start + 1
        while i < len(lines) and "|" in lines[i]:
            chunk.append(lines[i].rstrip("\n"))
            i += 1
        blocks.append(
            {
                "type": "table",
                "text": "\n".join(chunk),
                "start": start,
                "end": i,
                "path": path.copy(),
            }
        )
        return i

    def _parse_paragraph(
        self, lines: List[str], start: int, blocks: List[Dict], path: List[str]
    ) -> int:
        """Parse a paragraph."""
        chunk = [lines[start].rstrip("\n")]
        i = start + 1
        while (
            i < len(lines)
            and lines[i].strip() != ""
            and not any(
                [
                    self.FENCE_RE.match(lines[i].strip()),
                    re.match(r"^(#{1,6})\s+", lines[i]),
                    re.match(r"^\s*([-*+]|\d+\.)\s+", lines[i]),
                    lines[i].lstrip().startswith(">"),
                    re.match(r"^\s*([-*_])\s*(\1\s*){2,}$", lines[i]),
                    ("|" in lines[i] and re.match(r"^\s*\|", lines[i])),
                ]
            )
        ):
            chunk.append(lines[i].rstrip("\n"))
            i += 1
        blocks.append(
            {
                "type": "para",
                "text": "\n".join(chunk),
                "start": start,
                "end": i,
                "path": path.copy(),
            }
        )
        return i

    def _add_section_indices(self, blocks: List[Dict[str, Any]]) -> None:
        """Add per-section indices to blocks."""
        counters = {}
        for block in blocks:
            key = (tuple(block["path"]), block["type"])
            counters[key] = counters.get(key, 0) + 1
            block["idx_in_section"] = counters[key] - 1
