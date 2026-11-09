export type ColumnKind = "numeric" | "categorical" | "datetime" | "boolean";

export interface NumericSummary {
  count: number;
  mean: number | null;
  std: number | null;
  min: number | null;
  q25: number | null;
  median: number | null;
  q75: number | null;
  max: number | null;
}

export interface ColumnSummary {
  name: string;
  kind: ColumnKind;
  dtype: string;
  missing: number;
  missing_pct: number;
  unique: number;
  numeric: NumericSummary | null;
  top_values: { value: string; count: number }[];
  histogram: { bin: string; count: number }[];
}

export interface DatasetAnalysis {
  dataset_id?: string;
  dataset_available?: boolean;
  filename: string;
  size_bytes: number;
  row_count: number;
  column_count: number;
  total_missing: number;
  duplicate_rows: number;
  columns: ColumnSummary[];
  preview: Record<string, string | number | boolean | null>[];
}

export type MLTask = "auto" | "classification" | "regression";

export interface MLModelResult {
  name: string;
  metrics: Record<string, number>;
  confusion_matrix: number[][] | null;
  class_labels: (string | number | boolean | null)[];
  predictions: { actual: string | number | boolean | null; predicted: string | number | boolean | null }[];
  feature_importance: { feature: string; importance: number }[];
}

export interface MLInsights {
  provider: "rule_based" | "llm";
  label: string;
  title: string;
  summary: string;
  bullets: string[];
  limitations: string[];
  fallback_reason?: string;
}

export interface MLTrainingResult {
  filename: string;
  task: "classification" | "regression";
  target: string;
  random_seed: number;
  train_rows: number;
  test_rows: number;
  usable_rows: number;
  missing_target_rows: number;
  feature_count: number;
  features: string[];
  models: MLModelResult[];
  best_model: string;
  primary_metric: "f1_weighted" | "r2";
  class_labels: (string | number | boolean | null)[];
  insights: MLInsights;
  limitations: string[];
  training_seconds: number;
}

export type AnalystStatus = {
  state: "ready";
  provider: "deterministic_pandas";
  model: null;
  message: string;
  llm: { provider: "groq"; state: "missing_credentials" | "configured_unverified"; model: string | null; message: string };
};

export interface AgentCitation {
  tool: string;
  quote: string;
}

export interface AgentChatResult {
  status: "complete" | "unsupported";
  provider: "deterministic_pandas";
  model: null;
  answer: string;
  citations: AgentCitation[];
  tools_used: string[];
  tool_call_count: number;
  dataset: { filename: string; rows: number; columns: number };
  privacy: string;
}
