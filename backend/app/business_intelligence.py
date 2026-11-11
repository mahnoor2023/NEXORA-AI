"""Evidence-first business analytics for one explicitly mapped CSV dataset."""
from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

MAX_BI_ROWS = 100_000
MAX_BI_COLUMNS = 100
MAX_TOP_ROWS = 10

ROLE_ALIASES: dict[str, tuple[str, ...]] = {
    "order_id": ("order_id", "orderid", "order_number", "ordernumber", "invoice_id", "invoiceid", "purchase_id", "transaction_id"),
    "order_date": ("order_date", "orderdate", "purchase_date", "created_at", "createdat", "transaction_date", "date", "timestamp"),
    "product_id": ("product_id", "productid", "item_id", "sku", "product_sku"),
    "product_name": ("product_name", "product", "item_name", "description", "product_title"),
    "product_category": ("product_category", "category", "category_name", "product_type"),
    "quantity": ("quantity", "qty", "units", "item_quantity"),
    "unit_price": ("unit_price", "price", "item_price", "selling_price"),
    "line_total": ("line_total", "line_amount", "item_total", "extended_price", "subtotal"),
    "order_total": ("order_total", "order_amount", "total_amount", "grand_total", "total_price"),
    "revenue_amount": ("revenue", "revenue_amount", "sales_amount", "amount", "paid_amount", "payment_value"),
    "order_status": ("order_status", "status", "fulfillment_status", "payment_status"),
    "currency": ("currency", "currency_code", "currency_iso", "iso_currency"),
}

REVENUE_ROLES = ("line_total", "revenue_amount", "order_total", "quantity_x_unit_price")


class BIValidationError(ValueError):
    """Invalid or insufficient dataset mapping for trustworthy BI metrics."""


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def infer_mapping(columns: list[str]) -> dict[str, Any]:
    suggestions: dict[str, dict[str, Any]] = {}
    normalized = {_normalize_name(name): name for name in columns}
    used: set[str] = set()
    for role, aliases in ROLE_ALIASES.items():
        candidate = None
        confidence = "none"
        reason = "No clear header match; select a column manually if appropriate."
        for alias in aliases:
            key = _normalize_name(alias)
            if key in normalized and normalized[key] not in used:
                candidate = normalized[key]
                confidence = "high"
                reason = f"Header closely matches {alias.replace('_', ' ')}."
                break
        if candidate is None:
            soft = [name for name in columns if any(_normalize_name(alias) in _normalize_name(name) for alias in aliases) and name not in used]
            if len(soft) == 1:
                candidate = soft[0]
                confidence = "medium"
                reason = "Header contains a recognized business term; confirm the meaning."
        if candidate:
            used.add(candidate)
        suggestions[role] = {"column": candidate, "confidence": confidence, "reason": reason}

    qty = suggestions["quantity"]["column"]
    unit_price = suggestions["unit_price"]["column"]
    if suggestions["line_total"]["column"]:
        revenue = "line_total"
    elif suggestions["revenue_amount"]["column"]:
        revenue = "revenue_amount"
    elif suggestions["order_total"]["column"]:
        revenue = "order_total"
    elif qty and unit_price:
        revenue = "quantity_x_unit_price"
    else:
        revenue = None
    has_item_fields = bool(qty and (suggestions["product_id"]["column"] or suggestions["product_name"]["column"]))
    granularity = "order_item" if has_item_fields else "order"
    return {
        "suggestions": suggestions,
        "suggested_revenue_source": revenue,
        "suggested_granularity": granularity,
        "granularity_confidence": "medium" if has_item_fields else "low",
        "granularity_reason": "Product/quantity columns suggest item-level rows; confirm this before analysis." if has_item_fields else "Row granularity cannot be established from headers alone; choose one row per order or one row per order item.",
    }


def _parse_dates(values: pd.Series) -> pd.Series:
    # Parsing string forms prevents integers such as 20240101 being treated as
    # nanoseconds since epoch. Ambiguous/nonstandard formats remain invalid.
    strings = values.astype("string").str.strip().replace("", pd.NA)
    return pd.to_datetime(strings, errors="coerce", utc=True, format="mixed")


def _numeric(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    strings = values.astype("string").str.strip().str.replace(",", "", regex=False).str.replace("$", "", regex=False).str.replace("£", "", regex=False).str.replace("€", "", regex=False)
    parsed = pd.to_numeric(strings, errors="coerce")
    finite = pd.Series(np.isfinite(parsed.fillna(0).astype(float)), index=values.index)
    parsed = parsed.where(finite)
    invalid = values.notna() & parsed.isna()
    return parsed.astype(float), invalid


def _display(value: Any) -> str:
    return str(value)[:120]


def validate_mapping(mapping: dict[str, Any], columns: list[str]) -> None:
    if mapping.get("order_id") not in columns:
        raise BIValidationError("Map Order ID to a real column before calculating order metrics.")
    if mapping.get("granularity") not in {"order", "order_item"}:
        raise BIValidationError("Choose whether each row represents one order or one order item.")
    source = mapping.get("revenue_source")
    if source not in REVENUE_ROLES:
        raise BIValidationError("Select a revenue source or choose quantity multiplied by unit price.")
    required_by_source = {
        "line_total": "line_total",
        "revenue_amount": "revenue_amount",
        "order_total": "order_total",
    }
    if source in required_by_source and mapping.get(required_by_source[source]) not in columns:
        raise BIValidationError(f"The selected revenue source requires a valid {required_by_source[source].replace('_', ' ')} column.")
    if source == "quantity_x_unit_price" and (mapping.get("quantity") not in columns or mapping.get("unit_price") not in columns):
        raise BIValidationError("Revenue from quantity × unit price requires both mapped columns.")
    optional_roles = [role for role in ROLE_ALIASES if role != "order_id"]
    chosen = [mapping.get(role) for role in ("order_id", *optional_roles) if mapping.get(role)]
    invalid = [column for column in chosen if column not in columns]
    if invalid:
        raise BIValidationError("A selected mapping is not present in this dataset. Refresh the profile and map existing columns only.")
    duplicates = {column for column in chosen if chosen.count(column) > 1}
    if duplicates:
        raise BIValidationError("Map each dataset column to only one role. Conflicting selection: " + ", ".join(sorted(duplicates)) + ".")
    if source in required_by_source and mapping.get(required_by_source[source]) == mapping.get("order_id"):
        raise BIValidationError("Order ID cannot also be used as a revenue amount.")


def _revenue_entries(rows: pd.DataFrame, mapping: dict[str, Any], parsed_dates: pd.Series) -> tuple[pd.DataFrame, dict[str, int]]:
    source = mapping["revenue_source"]
    counts = {"unsupported_amount_rows": 0, "missing_amount_rows": 0, "invalid_numeric_amount_rows": 0, "negative_amount_rows": 0, "conflicting_order_total_ids": 0, "conflicting_order_date_ids": 0, "inconsistent_order_currency_ids": 0, "missing_currency_order_ids": 0}
    ids = rows["_bi_order_id"]
    currency_col = mapping.get("currency")
    currencies = rows[currency_col].astype("string").str.strip().str.upper() if currency_col else pd.Series(pd.NA, index=rows.index, dtype="string")
    currencies = currencies.mask(currencies == "", pd.NA)
    product_col = mapping.get("product_name") or mapping.get("product_id")
    product_values = rows[product_col].astype("string").str.strip() if product_col else pd.Series(pd.NA, index=rows.index, dtype="string")
    if mapping.get("product_name") and mapping.get("product_id"):
        ids_for_name = rows[mapping["product_id"]].astype("string").str.strip()
        product_values = product_values.mask(product_values.isna() | (product_values == ""), ids_for_name)
    category_col = mapping.get("product_category")
    category_values = rows[category_col].astype("string").str.strip() if category_col else pd.Series(pd.NA, index=rows.index, dtype="string")

    if source == "quantity_x_unit_price":
        quantity, invalid_quantity = _numeric(rows[mapping["quantity"]])
        unit_price, invalid_price = _numeric(rows[mapping["unit_price"]])
        amounts = quantity * unit_price
        invalid = invalid_quantity | invalid_price | quantity.isna() | unit_price.isna()
        counts["missing_amount_rows"] = int((quantity.isna() | unit_price.isna()).sum())
        counts["invalid_numeric_amount_rows"] = int((invalid_quantity | invalid_price).sum())
    else:
        amounts, invalid = _numeric(rows[mapping[source]])
        counts["missing_amount_rows"] = int(rows[mapping[source]].isna().sum())
        counts["invalid_numeric_amount_rows"] = int(invalid.sum())
    negative = amounts < 0
    counts["unsupported_amount_rows"] = int(invalid.sum())
    counts["negative_amount_rows"] = int(negative.sum())
    valid = amounts.notna() & ~negative
    raw = pd.DataFrame({
        "order_id": ids,
        "date": parsed_dates.loc[rows.index],
        "amount": amounts,
        "currency": currencies,
        "product": product_values,
        "category": category_values,
        "quantity": pd.to_numeric(rows[mapping["quantity"]], errors="coerce") if mapping.get("quantity") else np.nan,
        "valid": valid,
    }, index=rows.index)
    raw = raw.loc[raw["valid"]].drop(columns="valid")

    if source == "order_total" or mapping["granularity"] == "order":
        entries: list[dict[str, Any]] = []
        # Order totals repeated on every item are counted once. Conflicting
        # totals/currencies are excluded instead of silently choosing a value.
        for order_id, group in raw.groupby("order_id", sort=False, dropna=False):
            unique_amounts = group["amount"].dropna().unique()
            unique_currencies = group["currency"].dropna().unique()
            if currency_col and group["currency"].isna().any():
                counts["missing_currency_order_ids"] += 1
                continue
            if len(unique_amounts) != 1:
                counts["conflicting_order_total_ids"] += int(len(unique_amounts) > 1)
                continue
            if len(unique_currencies) > 1:
                counts["inconsistent_order_currency_ids"] += 1
                continue
            order_dates = group["date"].dropna().unique()
            if len(order_dates) > 1:
                counts["conflicting_order_date_ids"] += 1
            entries.append({
                "order_id": order_id,
                "date": order_dates[0] if len(order_dates) == 1 else pd.NaT,
                "amount": float(unique_amounts[0]),
                "currency": unique_currencies[0] if len(unique_currencies) else pd.NA,
                "product": group["product"].dropna().iloc[0] if group["product"].notna().any() else pd.NA,
                "category": group["category"].dropna().iloc[0] if group["category"].notna().any() else pd.NA,
                "quantity": group["quantity"].sum(min_count=1),
            })
        raw = pd.DataFrame(entries, columns=["order_id", "date", "amount", "currency", "product", "category", "quantity"])
    if raw.empty:
        raw = pd.DataFrame({
            "order_id": pd.Series(dtype="string"),
            "date": pd.Series(dtype="datetime64[ns, UTC]"),
            "amount": pd.Series(dtype="float64"),
            "currency": pd.Series(dtype="string"),
            "product": pd.Series(dtype="string"),
            "category": pd.Series(dtype="string"),
            "quantity": pd.Series(dtype="float64"),
        })
    return raw, counts


def _metric(value: Any, definition: str, columns: list[str], date_coverage: dict[str, Any], currency: str | None, excluded: dict[str, int], status: str = "available", reason: str | None = None) -> dict[str, Any]:
    return {"value": value, "status": status, "reason": reason, "definition": definition, "source_columns": columns, "date_coverage": date_coverage, "currency": currency, "excluded_records": excluded}


def compute_business_metrics(
    frame: pd.DataFrame,
    filename: str,
    mapping: dict[str, Any],
    start_date: date | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    if len(frame) > MAX_BI_ROWS or len(frame.columns) > MAX_BI_COLUMNS:
        raise BIValidationError(f"Business Intelligence is limited to {MAX_BI_ROWS:,} rows and {MAX_BI_COLUMNS} columns per dataset.")
    validate_mapping(mapping, [str(column) for column in frame.columns])
    if start_date and end_date and start_date > end_date:
        raise BIValidationError("Start date must be on or before end date.")
    if (start_date or end_date) and not mapping.get("order_date"):
        raise BIValidationError("Map an order date column before using a date range filter.")

    source_rows = len(frame)
    work = frame.copy()
    order_ids = work[mapping["order_id"]].astype("string").str.strip()
    valid_id = order_ids.notna() & (order_ids != "")
    status_col = mapping.get("order_status")
    status_excluded = pd.Series(False, index=work.index)
    if status_col:
        statuses = work[status_col].astype("string").str.casefold().str.replace("_", " ", regex=False)
        status_excluded = statuses.str.contains(r"\b(?:cancelled|canceled|cancel|voided|void|refunded|refund|returned|return|chargeback|failed)\b", regex=True, na=False)
        excluded_order_ids = set(order_ids.loc[status_excluded & valid_id].astype(str))
        if excluded_order_ids:
            status_excluded |= order_ids.astype(str).isin(excluded_order_ids)
    date_col = mapping.get("order_date")
    dates = _parse_dates(work[date_col]) if date_col else pd.Series(pd.NaT, index=work.index, dtype="datetime64[ns, UTC]")
    invalid_date = dates.isna() if date_col else pd.Series(False, index=work.index)

    in_range = pd.Series(True, index=work.index)
    if start_date:
        in_range &= dates >= pd.Timestamp(datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc))
    if end_date:
        end_exclusive = pd.Timestamp(datetime.combine(end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
        in_range &= dates < end_exclusive
    range_excluded = ~in_range if start_date or end_date else pd.Series(False, index=work.index)

    base_mask = valid_id & ~status_excluded
    usable_mask = base_mask & in_range
    base_usable = work.loc[base_mask].copy()
    usable = work.loc[usable_mask].copy()
    base_usable["_bi_order_id"] = order_ids.loc[base_usable.index]
    usable["_bi_order_id"] = order_ids.loc[usable.index]
    all_revenue_entries, _all_revenue_quality = _revenue_entries(base_usable, mapping, dates.loc[base_usable.index])
    usable_dates = dates.loc[usable.index]
    total_orders = int(usable["_bi_order_id"].nunique())
    duplicate_rows = int(frame.duplicated().sum())
    date_valid = dates.dropna()
    if len(date_valid):
        date_coverage = {"start": date_valid.min().date().isoformat(), "end": date_valid.max().date().isoformat(), "valid_date_rows": int(date_valid.size), "invalid_date_rows": int(invalid_date.sum())}
    else:
        date_coverage = {"start": None, "end": None, "valid_date_rows": 0, "invalid_date_rows": int(invalid_date.sum())}

    excluded = {
        "missing_order_id_rows": int((~valid_id).sum()),
        "cancelled_refunded_or_failed_rows": int(status_excluded.sum()),
        "outside_selected_date_range_rows": int(range_excluded.sum()),
        "invalid_date_rows": int(invalid_date.sum()),
        "exact_duplicate_rows_detected_not_automatically_removed": duplicate_rows,
    }
    revenue_entries, revenue_quality = _revenue_entries(usable, mapping, usable_dates)
    excluded.update(revenue_quality)
    currency_col = mapping.get("currency")
    unknown_currency = int(revenue_entries["currency"].isna().sum()) if not revenue_entries.empty else 0
    excluded["revenue_rows_missing_currency"] = unknown_currency if currency_col else 0
    if currency_col and not revenue_entries.empty:
        revenue_entries = revenue_entries.loc[revenue_entries["currency"].notna()].copy()
    currencies = sorted(str(value) for value in revenue_entries["currency"].dropna().unique()) if not revenue_entries.empty else []
    currency_mismatch = len(currencies) > 1
    currency = currencies[0] if len(currencies) == 1 else None
    source_role = mapping["revenue_source"]
    amount_columns = [mapping[source_role]] if source_role != "quantity_x_unit_price" else [mapping["quantity"], mapping["unit_price"]]
    if currency_mismatch:
        by_currency = revenue_entries.groupby("currency", sort=True)["amount"].sum().to_dict()
        revenue_value = None
        revenue_status = "unavailable"
        revenue_reason = "Multiple currencies are present; amounts are kept separate and no combined revenue is calculated."
        currency_breakdown = [{"currency": str(code), "revenue": round(float(amount), 6)} for code, amount in by_currency.items()]
    elif revenue_entries.empty:
        revenue_value = None
        revenue_status = "unavailable"
        revenue_reason = "No valid, non-negative revenue records remain after the selected status, date, amount, and order-ID rules."
        currency_breakdown = []
    else:
        revenue_value = round(float(revenue_entries["amount"].sum()), 6)
        revenue_status = "available"
        revenue_reason = None
        currency_breakdown = []
    eligible_revenue_orders = int(revenue_entries["order_id"].nunique()) if not currency_mismatch and not revenue_entries.empty else 0
    aov = round(revenue_value / eligible_revenue_orders, 6) if revenue_value is not None and eligible_revenue_orders else None

    revenue_metric = _metric(revenue_value, "Sum eligible non-negative amounts; repeated order totals are counted once per distinct order.", amount_columns, date_coverage, currency, excluded, revenue_status, revenue_reason)
    orders_metric = _metric(total_orders, "Count distinct nonblank order IDs after status exclusions and any selected date filter.", [mapping["order_id"]], date_coverage, None, excluded)
    aov_reason = "Revenue is unavailable or currencies differ." if aov is None else None
    aov_metric = _metric(aov, "Eligible total revenue divided by distinct orders with eligible revenue.", [mapping["order_id"], *amount_columns], date_coverage, currency, excluded, "available" if aov is not None else "unavailable", aov_reason)

    # Time series use mapped, valid dates only. Orders are deduplicated by ID in
    # each period; monetary trends stop when currencies could not be combined.
    dated_orders = pd.DataFrame({"order_id": order_ids, "date": dates}).loc[usable_mask & dates.notna()]
    monthly_orders: list[dict[str, Any]] = []
    weekly_orders: list[dict[str, Any]] = []
    if not dated_orders.empty:
        naive_dates = dated_orders["date"].dt.tz_convert(None)
        for period_name, periodizer, target in (("month", lambda s: s.dt.strftime("%Y-%m"), monthly_orders), ("week", lambda s: s.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d"), weekly_orders)):
            grouped = dated_orders.assign(period=periodizer(naive_dates)).groupby("period", sort=True)["order_id"].nunique()
            target.extend({"period": str(period), "orders": int(count)} for period, count in grouped.items())
    monthly_sales: list[dict[str, Any]] = []
    weekly_sales: list[dict[str, Any]] = []
    if not currency_mismatch and not revenue_entries.empty:
        dated_revenue = revenue_entries.loc[revenue_entries["date"].notna()].copy()
        if not dated_revenue.empty:
            dated_revenue["date"] = pd.to_datetime(dated_revenue["date"], utc=True)
            naive_revenue_dates = dated_revenue["date"].dt.tz_convert(None)
            for periodizer, target in ((lambda s: s.dt.strftime("%Y-%m"), monthly_sales), (lambda s: s.dt.to_period("W-SUN").dt.start_time.dt.strftime("%Y-%m-%d"), weekly_sales)):
                grouped = dated_revenue.assign(period=periodizer(naive_revenue_dates)).groupby("period", sort=True)["amount"].sum()
                target.extend({"period": str(period), "revenue": round(float(amount), 6), "currency": currency} for period, amount in grouped.items())

    product_col = mapping.get("product_name") or mapping.get("product_id")
    product_rows: list[dict[str, Any]] = []
    product_revenue_rows: list[dict[str, Any]] = []
    category_rows: list[dict[str, Any]] = []
    product_reason = "Map a product name or product ID to calculate product performance."
    category_reason = "Map a product category to calculate category performance."
    line_revenue_available = source_role != "order_total" and mapping["granularity"] == "order_item"
    if product_col and mapping.get("quantity"):
        quantity, _ = _numeric(usable[mapping["quantity"]])
        product_keys = usable[product_col].astype("string").str.strip()
        if mapping.get("product_name") and mapping.get("product_id"):
            product_keys = product_keys.mask(product_keys.isna() | (product_keys == ""), usable[mapping["product_id"]].astype("string").str.strip())
        product_valid = product_keys.notna() & (product_keys != "") & quantity.notna() & (quantity >= 0)
        product_data = pd.DataFrame({"product": product_keys.loc[product_valid], "quantity": quantity.loc[product_valid]})
        if line_revenue_available and not currency_mismatch and not revenue_entries.empty:
            # Revenue entries preserve source row indices for item-level sources.
            # Only include product allocations for line-based, item-grain data.
            revenue_column = mapping.get(source_role) if source_role != "quantity_x_unit_price" else None
            if revenue_column:
                item_amounts, _ = _numeric(usable[revenue_column])
            else:
                q, _ = _numeric(usable[mapping["quantity"]])
                p, _ = _numeric(usable[mapping["unit_price"]])
                item_amounts = q * p
            item_amounts = item_amounts.where(item_amounts >= 0)
            if currency_col:
                row_currency = usable[currency_col].astype("string").str.strip().str.upper()
                item_amounts = item_amounts.where(row_currency.notna() & row_currency.isin(currencies))
            product_data["revenue"] = item_amounts.loc[product_data.index].clip(lower=0)
        grouped = product_data.groupby("product", sort=False).agg(quantity=("quantity", "sum"), revenue=("revenue", "sum")) if "revenue" in product_data else product_data.groupby("product", sort=False).agg(quantity=("quantity", "sum"))
        quantity_ranked = grouped.sort_values("quantity", ascending=False, kind="stable").head(MAX_TOP_ROWS)
        product_rows = [{"product": _display(name), "quantity": _clean_metric(row["quantity"]), "revenue": _clean_metric(row["revenue"]) if "revenue" in row else None, "currency": currency if not currency_mismatch else None} for name, row in quantity_ranked.iterrows()]
        if "revenue" in grouped:
            revenue_ranked = grouped.sort_values("revenue", ascending=False, kind="stable").head(MAX_TOP_ROWS)
            product_revenue_rows = [{"product": _display(name), "quantity": _clean_metric(row["quantity"]), "revenue": _clean_metric(row["revenue"]), "currency": currency} for name, row in revenue_ranked.iterrows()]
        if line_revenue_available and not currency_mismatch and not revenue_entries.empty:
            product_reason = None
        elif currency_mismatch:
            product_reason = "Product revenue is not combined across currencies; quantity rankings are still available."
        elif source_role == "order_total":
            product_reason = "Order totals cannot be allocated reliably across products."
        else:
            product_reason = "Product revenue requires confirmed order-item granularity and a line-level revenue source."
    elif product_col:
        product_keys = usable[product_col].astype("string").str.strip()
        counts = product_keys[product_keys.notna() & (product_keys != "")].value_counts().head(MAX_TOP_ROWS)
        product_rows = [{"product": _display(name), "quantity": None, "revenue": None, "currency": currency} for name in counts.index]
        product_reason = "Map quantity to calculate product unit totals."

    if mapping.get("product_category"):
        category_keys = usable[mapping["product_category"]].astype("string").str.strip()
        category_valid = category_keys.notna() & (category_keys != "")
        category_data = pd.DataFrame({"category": category_keys.loc[category_valid], "order_id": usable.loc[category_valid, "_bi_order_id"]})
        category_group = category_data.groupby("category", sort=False).agg(orders=("order_id", "nunique"))
        if line_revenue_available and not currency_mismatch and not revenue_entries.empty:
            if source_role != "quantity_x_unit_price":
                category_amount, _ = _numeric(usable[mapping[source_role]])
            else:
                category_quantity, _ = _numeric(usable[mapping["quantity"]])
                category_price, _ = _numeric(usable[mapping["unit_price"]])
                category_amount = category_quantity * category_price
            category_amount = category_amount.where(category_amount >= 0)
            if currency_col:
                row_currency = usable[currency_col].astype("string").str.strip().str.upper()
                category_amount = category_amount.where(row_currency.notna() & row_currency.isin(currencies))
            category_amount = category_amount.where(category_keys.notna() & (category_keys != ""))
            # Aggregate against row-aligned data, then merge order contribution.
            categories_by_row = pd.DataFrame({"category": category_keys, "order_id": usable["_bi_order_id"], "amount": category_amount})
            category_revenue = categories_by_row.dropna(subset=["category", "amount"]).groupby("category", sort=False)["amount"].sum()
            category_group["revenue"] = category_revenue.reindex(category_group.index, fill_value=0)
        if "revenue" in category_group:
            category_group = category_group.sort_values("revenue", ascending=False, kind="stable").head(MAX_TOP_ROWS)
            category_reason = None
        else:
            category_group = category_group.sort_values("orders", ascending=False, kind="stable").head(MAX_TOP_ROWS)
            category_reason = "Category revenue requires confirmed order-item granularity, line-level amounts, and a single currency."
        category_rows = [{"category": _display(name), "orders": int(row["orders"]), "revenue": _clean_metric(row["revenue"]) if "revenue" in row else None, "order_contribution_pct": round(float(row["orders"]) / total_orders * 100, 2) if total_orders else None, "currency": currency if not currency_mismatch else None} for name, row in category_group.iterrows()]

    assumptions = [
        f"Confirmed granularity: one row per {'order' if mapping['granularity'] == 'order' else 'order item'}.",
        "Distinct order IDs are used for order counts. Repeated order totals are counted once per order; conflicting totals are excluded from revenue.",
        "If any row for an order has cancellation, void, refund, return, chargeback, or failed status text, all rows for that order are excluded from revenue and order KPIs.",
        "Negative amount records are excluded from gross revenue and reported as negative/refund candidates; refunds are not netted against sales.",
        "Exact duplicate rows are reported but not automatically removed because identical order-item rows can represent legitimate repeated items.",
        "Currency conversion is never performed. Currency is reported as unspecified when no currency field is mapped.",
    ]
    if not currency_col:
        assumptions.append("Currency was not supplied; monetary values are unlabelled and cannot be verified as one currency.")
    if start_date or end_date:
        assumptions.append("The selected date range is inclusive; rows with missing or invalid dates cannot match the range and are excluded.")

    comparison: dict[str, Any] = {"status": "unavailable", "reason": "Select both a start and end date to compare that period with the immediately preceding period of equal length."}
    if start_date and end_date and mapping.get("order_date") and not currency_mismatch:
        span_days = (end_date - start_date).days + 1
        previous_end = start_date - timedelta(days=1)
        previous_start = previous_end - timedelta(days=span_days - 1)
        coverage_start = date.fromisoformat(date_coverage["start"]) if date_coverage["start"] else None
        coverage_end = date.fromisoformat(date_coverage["end"]) if date_coverage["end"] else None
        if coverage_start and coverage_end and coverage_start <= previous_start and coverage_end >= end_date:
            # Compare observed sales for two exact equal-length windows. This
            # does not assert that source data is complete within either window.
            def window_entries(begin: date, finish: date) -> pd.DataFrame:
                begin_ts = pd.Timestamp(datetime.combine(begin, datetime.min.time(), tzinfo=timezone.utc))
                finish_ts = pd.Timestamp(datetime.combine(finish + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc))
                return all_revenue_entries.loc[all_revenue_entries["date"].ge(begin_ts) & all_revenue_entries["date"].lt(finish_ts)]
            current_period = window_entries(start_date, end_date)
            previous_period = window_entries(previous_start, previous_end)
            comparison_currencies = sorted(set(current_period["currency"].dropna().astype(str)) | set(previous_period["currency"].dropna().astype(str)))
            if current_period.empty or previous_period.empty:
                comparison = {"status": "unavailable", "reason": "At least one comparison period contains no eligible revenue records."}
            elif currency_col and (current_period["currency"].isna().any() or previous_period["currency"].isna().any()):
                comparison = {"status": "unavailable", "reason": "One comparison period contains records with missing currency."}
            elif len(comparison_currencies) > 1:
                comparison = {"status": "unavailable", "reason": "The comparison periods contain different currencies and are not combined."}
            else:
                current_value, prior_value = float(current_period["amount"].sum()), float(previous_period["amount"].sum())
                comparison = {"status": "available", "current_period": {"start": start_date.isoformat(), "end": end_date.isoformat(), "revenue": round(current_value, 6)}, "previous_period": {"start": previous_start.isoformat(), "end": previous_end.isoformat(), "revenue": round(prior_value, 6)}, "change": round(current_value - prior_value, 6), "change_pct": round((current_value - prior_value) / prior_value * 100, 2) if prior_value else None, "currency": comparison_currencies[0] if comparison_currencies else None, "coverage_completeness_verified": False}
        else:
            comparison = {"status": "unavailable", "reason": "The saved dataset does not cover both the selected date range and the immediately preceding equal-length period."}
    if currency_mismatch and start_date and end_date:
        comparison = {"status": "unavailable", "reason": "Period revenue comparison is unavailable because multiple currencies are present."}

    return {
        "filename": filename,
        "source_rows": source_rows,
        "selected_rows": int(usable_mask.sum()),
        "granularity": mapping["granularity"],
        "date_filter": {"start": start_date.isoformat() if start_date else None, "end": end_date.isoformat() if end_date else None},
        "metrics": {
            "revenue": revenue_metric,
            "orders": orders_metric,
            "average_order_value": aov_metric,
        },
        "currency_status": "multiple_currencies" if currency_mismatch else "single_currency" if currency else "not_supplied",
        "revenue_by_currency": currency_breakdown,
        "monthly_sales": monthly_sales,
        "weekly_sales": weekly_sales,
        "monthly_orders": monthly_orders,
        "weekly_orders": weekly_orders,
        "period_comparison": comparison,
        "top_products": {"items": product_rows, "by_revenue": product_revenue_rows, "source_column": product_col, "status": "available" if product_rows and product_reason is None else "partial" if product_rows else "unavailable", "reason": product_reason},
        "categories": {"items": category_rows, "source_column": mapping.get("product_category"), "status": "available" if category_rows and category_reason is None else "partial" if category_rows else "unavailable", "reason": category_reason},
        "data_quality": {
            "invalid_order_date_rows": int(invalid_date.sum()),
            "missing_order_id_rows": int((~valid_id).sum()),
            "status_excluded_rows": int(status_excluded.sum()),
            "exact_duplicate_rows_detected": duplicate_rows,
            "revenue_quality": revenue_quality,
            "product_key_missing_rows": int(usable[product_col].isna().sum()) if product_col else None,
            "currency_missing_revenue_rows": unknown_currency if currency_col else None,
        },
        "evidence": {
            "source_dataset": filename,
            "source_columns": {role: mapping.get(role) for role in ("order_id", "order_date", "product_id", "product_name", "product_category", "quantity", "unit_price", "line_total", "order_total", "revenue_amount", "order_status", "currency") if mapping.get(role)},
            "date_coverage": date_coverage,
            "assumptions": assumptions,
        },
    }


def _clean_metric(value: Any) -> float | int | None:
    if value is None or pd.isna(value) or not np.isfinite(float(value)):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else round(numeric, 6)
