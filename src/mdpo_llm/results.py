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


class Receipt(_DictLike):
    """Post-run usage/cost/duration summary.

    Cost fields are ``None`` when the model has no entry in
    ``litellm.model_cost``; renderers substitute ``"—"`` in that case.
    Prices are reported per 1M tokens (the rate card unit humans read)
    while totals are in USD.
    """

    def __init__(
        self,
        *,
        model: str,
        target_lang: str,
        source_path: Optional[str] = None,
        target_path: Optional[str] = None,
        po_path: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        api_calls: int = 0,
        duration_seconds: float = 0.0,
        input_cost_per_1m_usd: Optional[float] = None,
        output_cost_per_1m_usd: Optional[float] = None,
        input_cost_usd: Optional[float] = None,
        output_cost_usd: Optional[float] = None,
        total_cost_usd: Optional[float] = None,
    ):
        super().__init__(
            model=model,
            target_lang=target_lang,
            source_path=source_path,
            target_path=target_path,
            po_path=po_path,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            api_calls=api_calls,
            duration_seconds=duration_seconds,
            input_cost_per_1m_usd=input_cost_per_1m_usd,
            output_cost_per_1m_usd=output_cost_per_1m_usd,
            input_cost_usd=input_cost_usd,
            output_cost_usd=output_cost_usd,
            total_cost_usd=total_cost_usd,
        )

    def render(self, *, width: int = 60) -> str:
        """Return a human-readable multi-line receipt block."""
        bar = "=" * width
        sep = "-" * width

        def _price(v: Optional[float]) -> str:
            return "—" if v is None else f"${v:,.2f} / 1M tokens"

        def _cost(v: Optional[float]) -> str:
            return "—" if v is None else f"${v:,.6f}"

        def _path(label: str, value: Optional[str]) -> Optional[str]:
            if not value:
                return None
            return f"{label:<15} {value}"

        lines = [
            bar,
            "Translation receipt",
            sep,
            f"{'Model:':<15} {self['model']}",
            f"{'Target lang:':<15} {self['target_lang']}",
        ]
        for line in (
            _path("Source:", self["source_path"]),
            _path("Target:", self["target_path"]),
            _path("PO file:", self["po_path"]),
        ):
            if line is not None:
                lines.append(line)

        lines.extend(
            [
                sep,
                f"{'API calls:':<15} {self['api_calls']:,}",
                f"{'Input tokens:':<15} {self['input_tokens']:,}",
                f"{'Output tokens:':<15} {self['output_tokens']:,}",
                f"{'Total tokens:':<15} {self['total_tokens']:,}",
                sep,
                f"{'Input price:':<15} {_price(self['input_cost_per_1m_usd'])}",
                f"{'Output price:':<15} {_price(self['output_cost_per_1m_usd'])}",
                f"{'Input cost:':<15} {_cost(self['input_cost_usd'])}",
                f"{'Output cost:':<15} {_cost(self['output_cost_usd'])}",
                f"{'Total cost:':<15} {_cost(self['total_cost_usd'])}",
                sep,
                f"{'Wall clock:':<15} {self['duration_seconds']:.2f}s",
                bar,
            ]
        )
        return "\n".join(lines)


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
        receipt: Optional[Receipt] = None,
    ):
        super().__init__(
            source_path=source_path,
            target_path=target_path,
            po_path=po_path,
            blocks_count=blocks_count,
            coverage=coverage,
            translation_stats=translation_stats,
            receipt=receipt,
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
        receipt: Optional[Receipt] = None,
    ):
        super().__init__(
            source_dir=source_dir,
            target_dir=target_dir,
            po_dir=po_dir,
            files_processed=files_processed,
            files_failed=files_failed,
            files_skipped=files_skipped,
            results=list(results) if results else [],
            receipt=receipt,
        )
