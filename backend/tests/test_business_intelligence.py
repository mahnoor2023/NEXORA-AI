from datetime import date

import pandas as pd
import pytest

from app.business_intelligence import BIValidationError, compute_business_metrics, infer_mapping


def order_item_mapping(**overrides):
    mapping = {
        "order_id": "order_id", "order_date": "order_date", "product_id": "product_id",
        "product_name": "product_name", "product_category": "category", "quantity": "quantity",
        "unit_price": "unit_price", "line_total": "line_total", "order_total": None,
        "revenue_amount": None, "order_status": "status", "currency": "currency",
        "revenue_source": "line_total", "granularity": "order_item",
    }
    mapping.update(overrides)
    return mapping


def test_order_item_metrics_use_lines_and_count_distinct_orders():
    frame = pd.DataFrame([
        ["o1", "2025-01-02", "p1", "Tea", "Drink", 2, 4, 8, "paid", "USD"],
        ["o1", "2025-01-02", "p2", "Cup", "Home", 1, 3, 3, "paid", "USD"],
        ["o2", "2025-02-04", "p1", "Tea", "Drink", 1, 4, 4, "paid", "USD"],
    ], columns=["order_id", "order_date", "product_id", "product_name", "category", "quantity", "unit_price", "line_total", "status", "currency"])
    result = compute_business_metrics(frame, "orders.csv", order_item_mapping())
    assert result["metrics"]["revenue"]["value"] == 15
    assert result["metrics"]["orders"]["value"] == 2
    assert result["metrics"]["average_order_value"]["value"] == 7.5
    assert {item["product"]: item["quantity"] for item in result["top_products"]["items"]} == {"Tea": 3, "Cup": 1}
    assert sum(item["orders"] for item in result["categories"]["items"]) == 3


def test_repeated_order_total_is_counted_once_and_not_allocated_to_products():
    frame = pd.DataFrame([
        ["o1", "2025-01-01", "p1", "A", 1, 100, "USD"],
        ["o1", "2025-01-01", "p2", "B", 1, 100, "USD"],
        ["o2", "2025-01-02", "p3", "C", 1, 50, "USD"],
    ], columns=["order_id", "order_date", "product_id", "product_name", "quantity", "order_total", "currency"])
    mapping = order_item_mapping(order_total="order_total", line_total=None, revenue_source="order_total", order_status=None, unit_price=None, revenue_amount=None, product_category=None)
    result = compute_business_metrics(frame, "orders.csv", mapping)
    assert result["metrics"]["revenue"]["value"] == 150
    assert result["metrics"]["orders"]["value"] == 2
    assert result["metrics"]["average_order_value"]["value"] == 75
    assert result["top_products"]["items"][0]["revenue"] is None


def test_conflicting_order_totals_dates_and_duplicate_rows_are_reported():
    frame = pd.DataFrame([
        ["o1", "2025-01-01", 10, "USD"], ["o1", "2025-01-02", 12, "USD"],
        ["o2", "bad-date", 5, "USD"], ["o2", "bad-date", 5, "USD"],
    ], columns=["order_id", "order_date", "order_total", "currency"])
    mapping = order_item_mapping(order_date="order_date", order_total="order_total", line_total=None, product_id=None, product_name=None, product_category=None, quantity=None, revenue_source="order_total", granularity="order", order_status=None, unit_price=None)
    result = compute_business_metrics(frame, "orders.csv", mapping)
    assert result["metrics"]["revenue"]["value"] == 5
    assert result["data_quality"]["revenue_quality"]["conflicting_order_total_ids"] == 1
    assert result["data_quality"]["invalid_order_date_rows"] == 2
    assert result["data_quality"]["exact_duplicate_rows_detected"] == 1


def test_missing_and_invalid_amounts_negative_refunds_and_cancelled_orders_are_excluded():
    frame = pd.DataFrame([
        ["o1", "2025-01-01", "paid", "3", "USD"],
        ["o2", "2025-01-02", "paid", "bad", "USD"],
        ["o3", "2025-01-03", "paid", "-2", "USD"],
        ["o4", "2025-01-04", "cancelled", "7", "USD"],
        ["o4", "2025-01-04", "paid", "7", "USD"],
        ["o5", "bad", "paid", None, "USD"],
    ], columns=["order_id", "order_date", "status", "line_total", "currency"])
    mapping = order_item_mapping(order_date="order_date", product_id=None, product_name=None, product_category=None, quantity=None, unit_price=None, order_total=None, revenue_amount=None, currency=None)
    result = compute_business_metrics(frame, "orders.csv", mapping)
    assert result["metrics"]["revenue"]["value"] == 3
    assert result["metrics"]["orders"]["value"] == 4
    assert result["data_quality"]["status_excluded_rows"] == 2
    assert result["data_quality"]["revenue_quality"]["negative_amount_rows"] == 1
    assert result["data_quality"]["revenue_quality"]["invalid_numeric_amount_rows"] == 1
    assert result["data_quality"]["revenue_quality"]["missing_amount_rows"] == 1


def test_currency_mismatch_is_not_combined_and_missing_currency_is_excluded():
    frame = pd.DataFrame([
        ["o1", "2025-01-01", 10, "USD"], ["o2", "2025-01-02", 20, "EUR"],
        ["o3", "2025-01-03", 5, None],
    ], columns=["order_id", "order_date", "line_total", "currency"])
    result = compute_business_metrics(frame, "orders.csv", order_item_mapping(product_id=None, product_name=None, product_category=None, quantity=None, unit_price=None, order_status=None))
    assert result["metrics"]["revenue"]["value"] is None
    assert result["metrics"]["average_order_value"]["value"] is None
    assert result["currency_status"] == "multiple_currencies"
    assert result["revenue_by_currency"] == [{"currency": "EUR", "revenue": 20.0}, {"currency": "USD", "revenue": 10.0}]
    assert result["data_quality"]["currency_missing_revenue_rows"] == 1


def test_date_filter_is_inclusive_and_comparison_needs_both_periods():
    frame = pd.DataFrame([[f"o{i}", f"2025-01-{i:02}", 10, "USD"] for i in range(1, 7)], columns=["order_id", "order_date", "line_total", "currency"])
    mapping = order_item_mapping(product_id=None, product_name=None, product_category=None, quantity=None, unit_price=None, order_status=None)
    result = compute_business_metrics(frame, "orders.csv", mapping, date(2025, 1, 3), date(2025, 1, 5))
    assert result["metrics"]["revenue"]["value"] == 30
    assert result["metrics"]["orders"]["value"] == 3
    assert result["selected_rows"] == 3
    assert result["period_comparison"]["status"] == "unavailable"
    assert "does not cover" in result["period_comparison"]["reason"]
    with pytest.raises(BIValidationError, match="Start date"):
        compute_business_metrics(frame, "orders.csv", mapping, date(2025, 1, 5), date(2025, 1, 3))


def test_invalid_dates_and_missing_product_name_fall_back_to_product_id():
    frame = pd.DataFrame([["o1", "not-a-date", "p1", None, "x", 2, 4, 8, "paid", "USD"]], columns=["order_id", "order_date", "product_id", "product_name", "category", "quantity", "unit_price", "line_total", "status", "currency"])
    result = compute_business_metrics(frame, "orders.csv", order_item_mapping())
    assert result["top_products"]["items"][0]["product"] == "p1"
    assert result["data_quality"]["invalid_order_date_rows"] == 1
    assert result["monthly_sales"] == []


def test_empty_dataset_returns_unavailable_revenue_instead_of_failing():
    frame = pd.DataFrame(columns=["order_id", "line_total"])
    mapping = order_item_mapping(order_date=None, product_id=None, product_name=None, product_category=None, quantity=None, unit_price=None, order_total=None, revenue_amount=None, order_status=None, currency=None)
    result = compute_business_metrics(frame, "empty.csv", mapping)
    assert result["source_rows"] == 0
    assert result["metrics"]["revenue"]["value"] is None
    assert result["metrics"]["orders"]["value"] == 0


def test_incorrect_mapping_is_rejected():
    frame = pd.DataFrame({"order_id": ["o1"], "line_total": [5]})
    mapping = order_item_mapping(order_id="missing_id")
    with pytest.raises(BIValidationError, match="Order ID"):
        compute_business_metrics(frame, "orders.csv", mapping)


def test_inference_only_suggests_and_does_not_confirm_mapping():
    profile = infer_mapping(["order_id", "created_at", "sku", "quantity", "price"])
    assert profile["suggestions"]["order_id"]["column"] == "order_id"
    assert profile["suggested_revenue_source"] == "quantity_x_unit_price"
    assert profile["suggestions"]["line_total"]["column"] is None

