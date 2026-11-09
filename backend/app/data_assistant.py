"""Deterministic, read-only dataset answers computed directly with Pandas."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app.ml import MAX_ML_COLUMNS, MAX_ML_ROWS

MAX_TOP_N = 10


class UnsupportedQuestion(ValueError):
    pass


def _clean_number(value: Any) -> int | float | None:
    if pd.isna(value):
        return None
    result = float(value)
    if not np.isfinite(result):
        return None
    return int(result) if result.is_integer() else round(result, 6)


def _column_from(text: str, columns: list[str]) -> str | None:
    matches = [column for column in columns if re.search(rf"(?<!\w){re.escape(column)}(?!\w)", text, re.I)]
    return max(matches, key=len) if matches else None


def _numeric_columns(frame: pd.DataFrame) -> list[str]:
    return [str(name) for name in frame.columns if is_numeric_dtype(frame[name]) and not is_bool_dtype(frame[name])]


def _result(answer: str, tool: str, evidence: Any, filename: str, rows: int, columns: int, status: str = "complete") -> dict[str, Any]:
    return {
        "status": status,
        "provider": "deterministic_pandas",
        "model": None,
        "answer": answer,
        "citations": [{"tool": tool, "quote": str(evidence)}],
        "tools_used": [tool],
        "tool_call_count": 0,
        "dataset": {"filename": filename, "rows": rows, "columns": columns},
        "privacy": "Computed by Pandas in the NEXORA API. No LLM provider was used and no dataset values were sent to an AI provider.",
    }


def answer_dataset_question(question: str, frame: pd.DataFrame, filename: str) -> dict[str, Any]:
    """Answer supported requests using bounded operations; unsupported requests are explicit."""
    question = question.strip()
    if not question or len(question) > 1_000:
        raise ValueError("Question must contain 1 to 1,000 characters.")
    if len(frame) > MAX_ML_ROWS or len(frame.columns) > MAX_ML_COLUMNS:
        raise ValueError(f"Analysis is limited to {MAX_ML_ROWS:,} rows and {MAX_ML_COLUMNS} columns.")

    rows, count_columns = len(frame), len(frame.columns)
    columns = [str(name) for name in frame.columns]
    normalized = question.casefold()
    missing = frame.isna().sum()
    missing_total = int(missing.sum())

    if re.search(r"\b(column names?|fields?|schema)\b|\blist (the )?columns\b", normalized):
        evidence = {"columns": columns, "count": count_columns}
        return _result(f"The dataset has {count_columns} columns: {', '.join(columns)}.", "list_columns", evidence, filename, rows, count_columns)

    if re.search(r"\b(rows?|records?|observations?)\b", normalized) and re.search(r"\b(how many|count|number|total|size|large)\b", normalized):
        evidence = {"rows": rows}
        return _result(f"The dataset contains {rows:,} rows.", "count_rows", evidence, filename, rows, count_columns)

    if re.search(r"\b(missing|null|empty|blank|incomplete)\b", normalized):
        column = _column_from(question, columns)
        if column is not None:
            count = int(missing[column])
            percent = round(count / rows * 100, 2) if rows else 0
            evidence = {"column": column, "missing": count, "percent": percent}
            return _result(f"Column {column!r} has {count:,} missing values ({percent}% of {rows:,} rows).", "missing_values", evidence, filename, rows, count_columns)
        details = {str(name): int(value) for name, value in missing.items() if value}
        evidence = {"total_missing": missing_total, "columns_with_missing": details}
        if details:
            breakdown = "; ".join(f"{name}: {value:,}" for name, value in details.items())
            answer = f"There are {missing_total:,} missing values across {len(details)} columns: {breakdown}."
        else:
            answer = "There are no missing values in this dataset."
        return _result(answer, "missing_values", evidence, filename, rows, count_columns)

    top_match = re.search(r"\btop\s+(\d+)\b", normalized)
    if top_match and re.search(r"\b(group|groups|region|regions|category|categories|product|products|item|items|customer|customers|country|countries|segment|segments|rank|highest|most)\b", normalized):
        n = min(MAX_TOP_N, max(1, int(top_match.group(1))))
        metric = None
        by_match = re.search(r"\bby\s+(.+?)(?:\s+(?:for|in|from|across)\b|[?.!,]|$)", question, re.I)
        if by_match:
            metric = _column_from(by_match.group(1), columns)
            if metric is None and by_match.group(1).strip():
                raise UnsupportedQuestion(f"I could not match the requested ranking measure {by_match.group(1).strip()!r} to a dataset column.")
        group_text = question[:top_match.start()] + " " + (question[by_match.end():] if by_match else "")
        group_text = re.sub(r"\b(top|\d+|by|in|from|across|show|find|list|rank|highest|most|the|of)\b", " ", group_text, flags=re.I)
        group = _column_from(group_text, columns)
        if group is None:
            possible = [name for name in columns if name != metric and (not is_numeric_dtype(frame[name]) or is_bool_dtype(frame[name]))]
            if len(possible) == 1:
                group = possible[0]
        if group is None:
            raise UnsupportedQuestion("Name the column to group by, for example: ‘top 5 regions by revenue’.")
        if metric:
            if not is_numeric_dtype(frame[metric]) or is_bool_dtype(frame[metric]):
                raise UnsupportedQuestion(f"The ranking measure {metric!r} is not numeric. Choose a numeric measure or omit ‘by’ to count rows.")
            ranked = frame.groupby(group, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False, kind="stable").head(n)
            evidence = {"group_by": group, "measure": metric, "aggregation": "sum", "top": [{"value": str(key), "value_sum": _clean_number(value)} for key, value in ranked.items()]}
            labels = ", ".join(f"{key}: {_clean_number(value)}" for key, value in ranked.items()) or "no values"
            answer = f"Top {n} {group} groups by summed {metric}: {labels}."
        else:
            ranked = frame[group].fillna("(missing)").astype(str).value_counts().head(n)
            evidence = {"group_by": group, "aggregation": "row_count", "top": [{"value": str(key), "count": int(value)} for key, value in ranked.items()]}
            labels = ", ".join(f"{key}: {int(value)}" for key, value in ranked.items()) or "no values"
            answer = f"Top {n} {group} groups by row count: {labels}."
        return _result(answer, "top_groups", evidence, filename, rows, count_columns)

    frequency = re.search(r"\b(frequenc(?:y|ies)|value counts?|category counts?|distribution|breakdown|most common|categories)\b", normalized)
    if frequency:
        column = _column_from(question, columns)
        if column is None:
            raise UnsupportedQuestion("Name a column for category frequencies, for example: ‘show category counts for region’.")
        top = int(top_match.group(1)) if top_match else MAX_TOP_N
        top = min(MAX_TOP_N, max(1, top))
        values = frame[column].fillna("(missing)").astype(str).value_counts().head(top)
        evidence = {"column": column, "frequencies": [{"value": str(key), "count": int(value)} for key, value in values.items()]}
        labels = ", ".join(f"{key}: {int(value)}" for key, value in values.items()) or "no values"
        return _result(f"Most frequent values in {column}: {labels}.", "category_frequencies", evidence, filename, rows, count_columns)

    operation = next((name for name, pattern in (("mean", r"\b(mean|average|avg)\b"), ("median", r"\bmedian\b"), ("sum", r"\bsum\b"), ("minimum", r"\b(minimum|min)\b"), ("maximum", r"\b(maximum|max)\b"), ("standard deviation", r"\b(std|standard deviation)\b")) if re.search(pattern, normalized)), None)
    if operation:
        numeric = _numeric_columns(frame)
        column = _column_from(question, numeric)
        if column is None and len(numeric) == 1:
            column = numeric[0]
        if column is None:
            raise UnsupportedQuestion("Name one numeric column for this calculation. Available numeric columns: " + (", ".join(numeric) or "none") + ".")
        values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        functions = {"mean": values.mean, "median": values.median, "sum": values.sum, "minimum": values.min, "maximum": values.max, "standard deviation": values.std}
        value = _clean_number(functions[operation]()) if len(values) else None
        evidence = {"column": column, "operation": operation, "non_missing_numeric_values": int(len(values)), "result": value}
        if value is None:
            answer = f"Column {column!r} has no non-missing numeric values for this calculation."
        else:
            answer = f"The {operation} of {column!r} is {value} (computed from {len(values):,} non-missing numeric values)."
        return _result(answer, "numeric_aggregation", evidence, filename, rows, count_columns)

    if re.search(r"\b(summary|summarize|describe|overview|statistics)\b", normalized):
        evidence: dict[str, Any] = {"rows": rows, "columns": count_columns, "total_missing": missing_total, "columns_with_missing": int((missing > 0).sum()), "numeric_columns": {}}
        for column in _numeric_columns(frame)[:20]:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            evidence["numeric_columns"][column] = {"count": len(values), "mean": _clean_number(values.mean()), "median": _clean_number(values.median()), "min": _clean_number(values.min()), "max": _clean_number(values.max())}
        return _result(f"Dataset summary: {rows:,} rows, {count_columns} columns, {missing_total:,} missing values across {int((missing > 0).sum())} columns, and {len(_numeric_columns(frame))} numeric columns. Numeric summaries are available for: {', '.join(evidence['numeric_columns']) or 'none'}.", "dataset_summary", evidence, filename, rows, count_columns)

    raise UnsupportedQuestion("I can answer dataset row counts, column lists, missing values, summaries, category frequencies, numeric aggregations, and top-N groups. Please ask one of those questions and name the relevant column.")
