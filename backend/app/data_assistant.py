"""Deterministic, read-only dataset answers computed directly with Pandas."""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

from app.ml import MAX_ML_COLUMNS, MAX_ML_ROWS

MAX_TOP_N = 10
MAX_FILTER_RESULTS = 25


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
    normalized_text = text.casefold().replace("_", " ")
    matches: list[tuple[int, int, str]] = []
    for column in columns:
        normalized_column = column.casefold().replace("_", " ").strip()
        aliases = {(normalized_column, 3)}
        words = normalized_column.split()
        if words:
            aliases.add((" ".join(words[:-1] + [words[-1].removesuffix("s")]), 2))
            aliases.add((normalized_column + "s", 2))
        # A suffix such as `seller_state` can be referred to as “state” when
        # unique, but the lower score prevents it overriding an explicit name.
        if len(words) > 1:
            aliases.add((words[-1].removesuffix("s"), 1))
            aliases.add((words[-1].removesuffix("s") + "s", 1))
        for alias, score in aliases:
            alias = alias.strip()
            if alias and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_text, re.I):
                matches.append((score, len(alias), column))
    if not matches:
        return None
    best = max((score, length) for score, length, _ in matches)
    winners = {column for score, length, column in matches if (score, length) == best}
    return next(iter(winners)) if len(winners) == 1 else None


def _top_n(match: re.Match[str] | None) -> int:
    if match is None:
        return MAX_TOP_N
    requested = int(match.group(1))
    if not 1 <= requested <= MAX_TOP_N:
        raise UnsupportedQuestion(f"Top-N requests must be between 1 and {MAX_TOP_N}. Ask for a smaller number of results.")
    return requested


def _group_counts(frame: pd.DataFrame, group: str, entity_requested: bool) -> tuple[pd.Series, str | None]:
    """Count unique seller identifiers when available; otherwise count dataset rows."""
    identifier = None
    if entity_requested:
        identifiers = [str(name) for name in frame.columns if re.search(r"(?:^|_)seller_(?:id|identifier|key|code)$", str(name), re.I)]
        exact_ids = [name for name in identifiers if name.casefold() == "seller_id"]
        if len(exact_ids) == 1:
            identifier = exact_ids[0]
        elif len(identifiers) == 1:
            identifier = identifiers[0]
    if identifier is not None:
        counts = frame.groupby(group, dropna=False)[identifier].nunique(dropna=True)
    else:
        counts = frame[group].fillna("(missing)").astype(str).value_counts()
    return counts.sort_values(ascending=False, kind="stable"), identifier


def _count_filter(question: str) -> tuple[str, int] | None:
    match = re.search(
        r"\b(fewer\s+than|less\s+than|under|more\s+than|greater\s+than|over|at\s+least|at\s+most|exactly)\s+(-?\d+(?:\.\d+)?)\b",
        question,
        re.I,
    )
    if not match:
        return None
    phrase, raw_threshold = match.groups()
    try:
        numeric_threshold = float(raw_threshold)
    except ValueError:
        raise UnsupportedQuestion("Use a whole-number seller count from 0 to 20,000 for a group-count filter.") from None
    if not numeric_threshold.is_integer() or numeric_threshold < 0 or numeric_threshold > MAX_ML_ROWS:
        raise UnsupportedQuestion("Use a whole-number seller count from 0 to 20,000 for a group-count filter.")
    operator = {
        "fewer than": "lt", "less than": "lt", "under": "lt",
        "more than": "gt", "greater than": "gt", "over": "gt",
        "at least": "ge", "at most": "le", "exactly": "eq",
    }[re.sub(r"\s+", " ", phrase.casefold())]
    return operator, int(numeric_threshold)


def _safe_group_value(value: Any) -> str:
    return str(value)[:120]


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

    count_filter = _count_filter(question)
    if count_filter is not None:
        operator, threshold = count_filter
        group = _column_from(question, columns)
        if group is None:
            raise UnsupportedQuestion("I could not identify one grouping column for that count filter. Name the field, for example: ‘seller states with fewer than 5 sellers’.")
        entity_requested = bool(re.search(r"\bsellers?\b", question, re.I))
        counts, identifier = _group_counts(frame, group, entity_requested)
        comparisons = {"lt": counts.lt, "gt": counts.gt, "ge": counts.ge, "le": counts.le, "eq": counts.eq}
        filtered = counts.loc[comparisons[operator](threshold)]
        shown = filtered.head(MAX_FILTER_RESULTS)
        entity_label = f"distinct seller count ({identifier})" if identifier else "row count"
        operator_labels = {"lt": "below", "gt": "above", "ge": "at least", "le": "at most", "eq": "exactly"}
        operator_symbols = {"lt": "<", "gt": ">", "ge": ">=", "le": "<=", "eq": "="}
        evidence = {
            "group_by": group,
            "aggregation": "distinct_entity_count" if identifier else "row_count",
            "entity_column": identifier,
            "comparison": operator_symbols[operator],
            "threshold": threshold,
            "matching_group_count": int(len(filtered)),
            "groups": [{"value": _safe_group_value(value), "count": int(count)} for value, count in shown.items()],
            "groups_omitted": max(0, len(filtered) - len(shown)),
        }
        if filtered.empty:
            answer = f"No {group} groups have {entity_label} {operator_labels[operator]} {threshold}."
        else:
            labels = ", ".join(f"{_safe_group_value(value)} ({int(count)})" for value, count in shown.items())
            extra = f"; {len(filtered) - len(shown)} additional matching groups omitted" if len(filtered) > len(shown) else ""
            answer = f"{len(filtered)} {group} groups have {entity_label} {operator_labels[operator]} {threshold}: {labels}{extra}."
        return _result(answer, "filtered_group_counts", evidence, filename, rows, count_columns)

    top_match = re.search(r"\btop\s+(\d+)\b", normalized)
    count_by_match = re.search(r"\bcount\s+(?:of\s+)?(.+?)\s+by\s+(.+?)(?:[?.!,]|$)", question, re.I)
    most_pattern = re.search(r"\b(?:most|highest|largest)\s+(?:number\s+of\s+)?(?P<entity>[\w -]+?)(?:[?.!,]|$)", question, re.I)
    rank_request = top_match is not None or most_pattern is not None or count_by_match is not None
    if rank_request:
        n = _top_n(top_match)
        metric = None
        by_match = re.search(r"\bby\s+(.+?)(?:\s+(?:for|in|from|across)\b|[?.!,]|$)", question, re.I)
        if count_by_match:
            group = _column_from(count_by_match.group(2), columns)
            if group is None:
                raise UnsupportedQuestion("I could not match the grouping field to one dataset column. Use its column name or a unique human-friendly name.")
            counts, identifier = _group_counts(frame, group, bool(re.search(r"seller", count_by_match.group(1), re.I)))
            evidence = {"group_by": group, "aggregation": "distinct_entity_count" if identifier else "row_count", "entity_column": identifier, "top": [{"value": str(key), "count": int(value)} for key, value in counts.head(n).items()]}
            labels = ", ".join(f"{item['value']}: {item['count']}" for item in evidence["top"]) or "no values"
            measure = f"distinct sellers ({identifier})" if identifier else "rows"
            return _result(f"Counts by {group} ({measure}): {labels}.", "grouped_entity_counts", evidence, filename, rows, count_columns)
        if by_match and not re.fullmatch(r"(?:number\s+of\s+)?(?:sellers?|customers?|users?|orders?|records?|rows?|items?|products?|transactions?)", by_match.group(1).strip(), re.I):
            metric = _column_from(by_match.group(1), columns)
            if metric is None and by_match.group(1).strip():
                if not re.search(r"\b(number\s+of|count\s+of|count|sellers?|customers?|users?|orders?|records?|rows?|items?|products?|transactions?)\b", by_match.group(1), re.I):
                    raise UnsupportedQuestion(f"I could not match the requested ranking measure {by_match.group(1).strip()!r} to a dataset column.")
        if top_match:
            group_end = by_match.start() if by_match else len(question)
            group_text = question[top_match.end():group_end]
        elif most_pattern:
            group_text = question[:most_pattern.start("entity")] + " " + question[most_pattern.end("entity"):]
            group_text += " " + most_pattern.group("entity")
        else:
            group_text = question[:count_by_match.start(1)]
        group_text = re.sub(r"\b(top|\d+|by|in|from|across|show|find|list|rank|highest|most|the|of|number|count|how many|sellers?|customers?|users?|orders?|records?|rows?|items?|products?|transactions?)\b", " ", group_text, flags=re.I)
        group = _column_from(group_text, columns)
        # “Which states have the most sellers?” commonly names the entity and
        # dimension separately; use the unique matching state field in that form.
        if group is None and most_pattern and re.search(r"\b(state|states)\b", question, re.I):
            state_columns = [name for name in columns if re.search(r"(?:^|_)state$", name, re.I)]
            if len(state_columns) == 1:
                group = state_columns[0]
        if group is None:
            possible = [name for name in columns if name != metric and (not is_numeric_dtype(frame[name]) or is_bool_dtype(frame[name]))]
            if len(possible) == 1:
                group = possible[0]
        if group is None:
            raise UnsupportedQuestion("I could not identify one grouping column. Name the dataset field, for example: ‘top 5 seller states by number of sellers’.")
        if metric:
            if not is_numeric_dtype(frame[metric]) or is_bool_dtype(frame[metric]):
                raise UnsupportedQuestion(f"The ranking measure {metric!r} is not numeric. Choose a numeric measure or omit ‘by’ to count rows.")
            ranked = frame.groupby(group, dropna=False)[metric].sum(min_count=1).sort_values(ascending=False, kind="stable").head(n)
            evidence = {"group_by": group, "measure": metric, "aggregation": "sum", "top": [{"value": str(key), "value_sum": _clean_number(value)} for key, value in ranked.items()]}
            labels = ", ".join(f"{key}: {_clean_number(value)}" for key, value in ranked.items()) or "no values"
            answer = f"Top {n} {group} groups by summed {metric}: {labels}."
        else:
            entity_requested = bool(re.search(r"seller", question, re.I))
            ranked, identifier = _group_counts(frame, group, entity_requested)
            ranked = ranked.head(n)
            evidence = {"group_by": group, "aggregation": "distinct_entity_count" if identifier else "row_count", "entity_column": identifier, "top": [{"value": str(key), "count": int(value)} for key, value in ranked.items()]}
            labels = ", ".join(f"{key}: {int(value)}" for key, value in ranked.items()) or "no values"
            measure = f"distinct sellers ({identifier})" if identifier else "row count"
            answer = f"Top {n} {group} groups by {measure}: {labels}."
        return _result(answer, "top_groups", evidence, filename, rows, count_columns)

    frequency = re.search(r"\b(frequenc(?:y|ies)|value counts?|category counts?|distribution|breakdown|most common|categories)\b", normalized)
    if frequency:
        column = _column_from(question, columns)
        if column is None:
            raise UnsupportedQuestion("Name a column for category frequencies, for example: ‘show category counts for region’.")
        top = _top_n(top_match) if top_match else MAX_TOP_N
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
