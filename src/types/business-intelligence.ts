export type BIRole = "order_id" | "order_date" | "product_id" | "product_name" | "product_category" | "quantity" | "unit_price" | "line_total" | "order_total" | "revenue_amount" | "order_status" | "currency";

export type BIMapping = Record<BIRole, string | null> & {
  order_id: string;
  revenue_source: "line_total" | "revenue_amount" | "order_total" | "quantity_x_unit_price";
  granularity: "order" | "order_item";
};

export type BIProfile = {
  dataset_id: string;
  filename: string;
  row_count: number;
  column_count: number;
  columns: { name: string; dtype: string; missing: number }[];
  mapping_inference: {
    suggestions: Record<BIRole, { column: string | null; confidence: string; reason: string }>;
    suggested_revenue_source: BIMapping["revenue_source"] | null;
    suggested_granularity: BIMapping["granularity"];
    granularity_confidence: string;
    granularity_reason: string;
  };
  mapping_confirmed: false;
};

export type BIMetric = {
  value: number | null;
  status: "available" | "unavailable";
  reason: string | null;
  definition: string;
  source_columns: string[];
  date_coverage: { start: string | null; end: string | null; valid_date_rows: number; invalid_date_rows: number };
  currency: string | null;
  excluded_records: Record<string, number>;
};

export type BIResult = {
  filename: string;
  source_rows: number;
  selected_rows: number;
  granularity: string;
  date_filter: { start: string | null; end: string | null };
  metrics: { revenue: BIMetric; orders: BIMetric; average_order_value: BIMetric };
  currency_status: string;
  revenue_by_currency: { currency: string; revenue: number }[];
  monthly_sales: { period: string; revenue: number; currency: string | null }[];
  weekly_sales: { period: string; revenue: number; currency: string | null }[];
  monthly_orders: { period: string; orders: number }[];
  weekly_orders: { period: string; orders: number }[];
  period_comparison: { status: string; reason?: string; change_pct?: number | null; currency?: string | null };
  top_products: { items: { product: string; quantity: number | null; revenue: number | null; currency: string | null }[]; by_revenue: { product: string; quantity: number | null; revenue: number | null; currency: string | null }[]; status: string; reason: string | null; source_column: string | null };
  categories: { items: { category: string; orders: number; revenue: number | null; order_contribution_pct: number | null; currency: string | null }[]; status: string; reason: string | null; source_column: string | null };
  data_quality: Record<string, unknown>;
  evidence: { source_dataset: string; source_columns: Record<string, string>; date_coverage: BIMetric["date_coverage"]; assumptions: string[] };
};
