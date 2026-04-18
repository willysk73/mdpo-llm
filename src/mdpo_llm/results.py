"""
Typed result dataclasses.

Frozen dataclasses with a ``Mapping``-like shim so existing dict-style
consumers (``result["coverage"]`` / ``result.get("translation_stats", {})``)
continue to work unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List, Optional, Union


class _DictLike(dict):
    """Base for typed results that stay fully dict-compatible.

    Subclassing ``dict`` keeps ``json.dumps(result)`` and any other code that
    type-checks for ``dict`` working (subclassing only
    ``collections.abc.Mapping`` broke that).  We layer attribute access on
    top so consumers can use either ``result["x"]`` or ``result.x``.  The
    class is still a real ``Mapping`` via ``dict``'s registration, so
    ``isinstance(result, Mapping)`` holds.
    """

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain ``dict`` copy, recursively unwrapping nested result types."""
        out: Dict[str, Any] = {}
        for key, value in self.items():
            if isinstance(value, _DictLike):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [
                    v.to_dict() if isinstance(v, _DictLike) else v for v in value
                ]
            else:
                out[key] = value
        return out


class Coverage(_DictLike):
    def __init__(
        self,
        *,
        total_blocks: int,
        translatable_blocks: int,
        translated_blocks: int,
        fuzzy_blocks: int,
        untranslated_blocks: int,
        coverage_percentage: float,
        by_type: Dict[str, Dict[str, int]],
    ):
        super().__init__(
            total_blocks=total_blocks,
            translatable_blocks=translatable_blocks,
            translated_blocks=translated_blocks,
            fuzzy_blocks=fuzzy_blocks,
            untranslated_blocks=untranslated_blocks,
            coverage_percentage=coverage_percentage,
            by_type=by_type,
        )


class BatchStats(_DictLike):
    def __init__(
        self,
        *,
        processed: int = 0,
        failed: int = 0,
        skipped: int = 0,
        batched_calls: int = 0,
        per_entry_calls: int = 0,
        validated: int = 0,
        validation_failed: int = 0,
    ):
        super().__init__(
            processed=processed,
            failed=failed,
            skipped=skipped,
            batched_calls=batched_calls,
            per_entry_calls=per_entry_calls,
            validated=validated,
            validation_failed=validation_failed,
        )


class ProcessResult(_DictLike):
    def __init__(
        self,
        *,
        source_path: str,
        target_path: str,
        po_path: str,
        blocks_count: int,
        coverage: Coverage,
        translation_stats: BatchStats,
    ):
        super().__init__(
            source_path=source_path,
            target_path=target_path,
            po_path=po_path,
            blocks_count=blocks_count,
            coverage=coverage,
            translation_stats=translation_stats,
        )


class DirectoryResult(_DictLike):
    def __init__(
        self,
        *,
        source_dir: str,
        target_dir: str,
        po_dir: Optional[str],
        files_processed: int,
        files_failed: int,
        files_skipped: int,
        results: Optional[List[Union[ProcessResult, Dict[str, Any]]]] = None,
    ):
        super().__init__(
            source_dir=source_dir,
            target_dir=target_dir,
            po_dir=po_dir,
            files_processed=files_processed,
            files_failed=files_failed,
            files_skipped=files_skipped,
            results=list(results) if results else [],
        )
