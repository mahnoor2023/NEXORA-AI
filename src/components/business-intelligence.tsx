"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AlertCircle, BarChart3, CheckCircle2, ChevronDown, LoaderCircle, RefreshCw, TrendingUp } from "lucide-react";
import { api, type DatasetRecord } from "@/lib/auth";
import { analyzeBI, getBIProfile } from "@/lib/business-intelligence";
import type { BIProfile, BIResult, BIMapping, BIRole } from "@/types/business-intelligence";

const roleLabels: Record<BIRole, string> = { order_id: "Order ID (required)", order_date: "Order date", product_id: "Product ID / SKU", product_name: "Product name", product_category: "Product category", quantity: "Quantity", unit_price: "Unit price", line_total: "Line total", order_total: "Order total", revenue_amount: "Revenue / payment amount", order_status: "Order status", currency: "Currency" };
const roles = Object.keys(roleLabels) as BIRole[];
const blankMapping = (): BIMapping => ({ order_id: "", order_date: null, product_id: null, product_name: null, product_category: null, quantity: null, unit_price: null, line_total: null, order_total: null, revenue_amount: null, order_status: null, currency: null, revenue_source: "line_total", granularity: "order_item" });
const inputClass = "mt-1 h-10 w-full rounded-lg border border-[#e2e6ea] bg-white px-3 text-sm text-[#384351] outline-none focus:border-[#8b7ee9]";
const panelClass = "rounded-xl border border-[#e8ebef] bg-white p-5";
const accent = "#7162dc";

function valueText(value: number | null, metric: BIResult["metrics"]["revenue"], money = false) {
  if (value === null) return "Unavailable";
  const formatted = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(value);
  return money ? `${metric.currency ? `${metric.currency} ` : ""}${formatted}` : formatted;
}

function Kpi({ title, metric, money = false }: { title: string; metric: BIResult["metrics"]["revenue"]; money?: boolean }) {
  return <article className={panelClass}><p className="text-xs font-semibold uppercase tracking-wide text-[#8b95a0]">{title}</p><p className="mt-3 text-2xl font-bold text-[#303945]">{valueText(metric.value, metric, money)}</p><p className="mt-2 text-xs leading-5 text-[#7f8995]">{metric.definition}</p>{metric.reason && <p className="mt-2 flex gap-1.5 text-xs text-[#a56b43]"><AlertCircle size={14} className="mt-0.5 shrink-0"/>{metric.reason}</p>}<p className="mt-2 text-[10px] text-[#9aa2ac]">Source: {metric.source_columns.join(", ") || "none"} · {metric.currency ?? "currency not specified"}</p></article>;
}

export function BusinessIntelligence() {
  const [datasets, setDatasets] = useState<DatasetRecord[]>([]);
  const [datasetId, setDatasetId] = useState("");
  const [profile, setProfile] = useState<BIProfile | null>(null);
  const [mapping, setMapping] = useState<BIMapping>(blankMapping());
  const [result, setResult] = useState<BIResult | null>(null);
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [trend, setTrend] = useState<"month" | "week">("month");
  const [productRanking, setProductRanking] = useState<"quantity" | "revenue">("quantity");
  const [loadingDatasets, setLoadingDatasets] = useState(true);
  const [loadingProfile, setLoadingProfile] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    api<{ datasets: DatasetRecord[] }>("/datasets").then(({ datasets: records }) => {
      if (!active) return;
      const available = records.filter((record) => record.available);
      setDatasets(available);
      const requested = new URLSearchParams(window.location.search).get("dataset");
      setDatasetId(available.some((record) => record.id === requested) ? requested! : available[0]?.id ?? "");
    }).catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Could not load your datasets."); }).finally(() => { if (active) setLoadingDatasets(false); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!datasetId) { setProfile(null); setResult(null); return; }
    let active = true;
    setLoadingProfile(true); setError(""); setProfile(null); setResult(null);
    getBIProfile(datasetId).then((value) => {
      if (!active) return;
      setProfile(value);
      const suggested = blankMapping();
      for (const role of roles) {
        const column = value.mapping_inference.suggestions[role].column;
        if (role === "order_id") suggested.order_id = column ?? "";
        else suggested[role] = column;
      }
      suggested.order_id = value.mapping_inference.suggestions.order_id.column ?? "";
      suggested.revenue_source = value.mapping_inference.suggested_revenue_source ?? "line_total";
      suggested.granularity = value.mapping_inference.suggested_granularity;
      setMapping(suggested);
    }).catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Could not inspect this dataset."); }).finally(() => { if (active) setLoadingProfile(false); });
    return () => { active = false; };
  }, [datasetId]);

  const trendData = useMemo(() => {
    if (!result) return [];
    const sales = trend === "month" ? result.monthly_sales : result.weekly_sales;
    const orders = trend === "month" ? result.monthly_orders : result.weekly_orders;
    const indexed = new Map<string, { period: string; revenue?: number; orders?: number }>(sales.map((item) => [item.period, { period: item.period, revenue: item.revenue }]));
    for (const item of orders) indexed.set(item.period, { ...indexed.get(item.period), period: item.period, orders: item.orders });
    return [...indexed.values()].sort((left, right) => left.period.localeCompare(right.period));
  }, [result, trend]);
  const productItems = productRanking === "revenue" ? result?.top_products.by_revenue ?? [] : result?.top_products.items ?? [];

  const setRole = (role: BIRole, value: string) => setMapping((current) => ({ ...current, [role]: value || null, ...(role === "order_id" ? { order_id: value } : {}) }));
  const calculate = async () => {
    if (!datasetId) return;
    setError(""); setAnalyzing(true);
    try { const next = await analyzeBI(datasetId, mapping, startDate, endDate); setResult(next); }
    catch (cause) { setResult(null); setError(cause instanceof Error ? cause.message : "Business Intelligence analysis failed."); }
    finally { setAnalyzing(false); }
  };

  return <main className="mx-auto max-w-7xl px-5 py-7 pb-24 sm:px-8 md:pb-10">
    <div className="flex flex-wrap items-start justify-between gap-4"><div><div className="flex items-center gap-2 text-sm font-semibold text-[#6c60d7]"><TrendingUp size={17}/>Business intelligence</div><h1 className="mt-2 text-2xl font-bold text-[#27313d] sm:text-3xl">Sales performance</h1><p className="mt-1 max-w-3xl text-sm leading-6 text-[#83909c]">Deterministic metrics from your saved CSV. Confirm each mapping and row granularity before computing. No sales data or LLM-generated estimates are added.</p></div><Link href="/new-analysis" className="rounded-lg border border-[#e3e6eb] bg-white px-3 py-2 text-xs font-semibold text-[#596574]">Choose another dataset</Link></div>

    {error && <p role="alert" className="mt-5 flex items-start gap-2 rounded-lg border border-[#f1d8d5] bg-[#fff6f5] p-3 text-sm text-[#a54f4b]"><AlertCircle className="mt-0.5 shrink-0" size={16}/>{error}</p>}
    {loadingDatasets ? <div className="mt-8 flex items-center gap-2 text-sm text-[#87919d]"><LoaderCircle className="animate-spin" size={16}/>Loading your saved datasets…</div> : datasets.length === 0 ? <section className={`${panelClass} mt-6 text-center`}><BarChart3 className="mx-auto text-[#9389e4]"/><h2 className="mt-3 font-semibold text-[#3b4653]">No saved CSV is available</h2><p className="mt-1 text-sm text-[#85909c]">Upload a dataset with appropriate business columns to start. No metrics can be calculated yet.</p><Link href="/new-analysis" className="mt-4 inline-block rounded-lg bg-[#6f61df] px-4 py-2 text-sm font-semibold text-white">Upload dataset</Link></section> : <>
      <section className={`${panelClass} mt-6 grid gap-4 md:grid-cols-[minmax(200px,1fr)_auto] md:items-end`}><label className="text-xs font-semibold text-[#596574]">Saved dataset<select aria-label="Saved dataset" className={inputClass} value={datasetId} onChange={(event) => setDatasetId(event.target.value)}>{datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.filename}</option>)}</select></label>{profile && <p className="text-xs text-[#85909c]">{profile.row_count.toLocaleString()} rows · {profile.column_count} columns · mapping requires confirmation</p>}</section>

      {loadingProfile ? <div className="mt-5 flex items-center gap-2 text-sm text-[#87919d]"><LoaderCircle className="animate-spin" size={16}/>Inspecting dataset columns…</div> : profile && <>
        <section className={`${panelClass} mt-5`}><div className="flex flex-wrap items-start justify-between gap-3"><div><h2 className="font-semibold text-[#303945]">Confirm business column mappings</h2><p className="mt-1 text-xs leading-5 text-[#85909c]">Suggestions are based only on headers. Check their meaning against your data. CSVs without suitable sales fields will not receive sales KPIs.</p></div><span className="rounded-full bg-[#fff7e9] px-3 py-1 text-[10px] font-semibold text-[#9c7138]">Not confirmed</span></div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">{roles.map((role) => <label key={role} className="text-xs font-semibold text-[#5b6673]">{roleLabels[role]}<select aria-label={roleLabels[role]} className={inputClass} value={mapping[role] ?? ""} onChange={(event) => setRole(role, event.target.value)}><option value="">Not mapped</option>{profile.columns.map((column) => <option key={column.name} value={column.name}>{column.name} · {column.dtype}{column.missing ? ` · ${column.missing} missing` : ""}</option>)}</select><span className="mt-1 block text-[10px] font-normal text-[#98a0aa]">{profile.mapping_inference.suggestions[role].reason}</span></label>)}</div>
          <div className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4"><label className="text-xs font-semibold text-[#5b6673]">CSV row granularity<select className={inputClass} value={mapping.granularity} onChange={(event) => setMapping((current) => ({ ...current, granularity: event.target.value as BIMapping["granularity"] }))}><option value="order">One row per order</option><option value="order_item">One row per order item</option></select><span className="mt-1 block text-[10px] font-normal text-[#98a0aa]">{profile.mapping_inference.granularity_reason}</span></label><label className="text-xs font-semibold text-[#5b6673]">Revenue calculation<select className={inputClass} value={mapping.revenue_source} onChange={(event) => setMapping((current) => ({ ...current, revenue_source: event.target.value as BIMapping["revenue_source"] }))}><option value="line_total">Line total</option><option value="revenue_amount">Revenue / payment amount</option><option value="order_total">Order total (deduplicate by order)</option><option value="quantity_x_unit_price">Quantity × unit price</option></select><span className="mt-1 block text-[10px] font-normal text-[#98a0aa]">Order total values must be consistent within each order.</span></label><label className="text-xs font-semibold text-[#5b6673]">Start date (inclusive)<input aria-label="Start date" type="date" className={inputClass} value={startDate} onChange={(event) => setStartDate(event.target.value)}/></label><label className="text-xs font-semibold text-[#5b6673]">End date (inclusive)<input aria-label="End date" type="date" className={inputClass} value={endDate} onChange={(event) => setEndDate(event.target.value)}/></label></div>
          {mapping.revenue_source === "line_total" && !mapping.line_total && <p className="mt-3 rounded-lg bg-[#fff8ec] p-3 text-xs text-[#956b35]">Select a line total column or choose a revenue source that matches your CSV.</p>}{mapping.revenue_source === "revenue_amount" && !mapping.revenue_amount && <p className="mt-3 rounded-lg bg-[#fff8ec] p-3 text-xs text-[#956b35]">Select a revenue amount column to continue.</p>}{mapping.revenue_source === "order_total" && !mapping.order_total && <p className="mt-3 rounded-lg bg-[#fff8ec] p-3 text-xs text-[#956b35]">Select an order total column to continue.</p>}{mapping.revenue_source === "quantity_x_unit_price" && (!mapping.quantity || !mapping.unit_price) && <p className="mt-3 rounded-lg bg-[#fff8ec] p-3 text-xs text-[#956b35]">Map both quantity and unit price to calculate revenue.</p>}
          <div className="mt-4 flex flex-wrap items-center gap-3"><button disabled={analyzing || !mapping.order_id || (startDate && endDate ? startDate > endDate : false) || (mapping.revenue_source === "line_total" && !mapping.line_total) || (mapping.revenue_source === "revenue_amount" && !mapping.revenue_amount) || (mapping.revenue_source === "order_total" && !mapping.order_total) || (mapping.revenue_source === "quantity_x_unit_price" && (!mapping.quantity || !mapping.unit_price))} onClick={() => void calculate()} className="inline-flex items-center gap-2 rounded-lg bg-[#6f61df] px-4 py-2.5 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:opacity-50">{analyzing ? <LoaderCircle size={15} className="animate-spin"/> : <CheckCircle2 size={15}/>} {analyzing ? "Calculating from CSV…" : "Confirm mappings and calculate"}</button>{result && <button onClick={() => void calculate()} className="inline-flex items-center gap-1.5 rounded-lg border border-[#e1e5ea] px-3 py-2 text-xs font-semibold text-[#626e7b]"><RefreshCw size={13}/>Recalculate</button>}{startDate && endDate && startDate > endDate && <span role="alert" className="text-xs text-[#ae5252]">Start date must be on or before end date.</span>}</div>
        </section>

        {result && <>
          <div className="mt-5 grid gap-4 md:grid-cols-3"><Kpi title="Total revenue" metric={result.metrics.revenue} money/><Kpi title="Distinct orders" metric={result.metrics.orders}/><Kpi title="Average order value" metric={result.metrics.average_order_value} money/></div>
          <div className="mt-4 grid gap-4 xl:grid-cols-[1.25fr_1fr]"><section className={panelClass}><div className="flex flex-wrap items-start justify-between gap-2"><div><h2 className="font-semibold text-[#303945]">Sales trend</h2><p className="mt-1 text-xs text-[#89939f]">Revenue only appears when it can be computed in one currency.</p></div><div className="flex gap-1 rounded-lg bg-[#f5f6f8] p-1"><button onClick={() => setTrend("month")} aria-pressed={trend === "month"} className={`rounded px-2.5 py-1.5 text-xs ${trend === "month" ? "bg-white font-semibold text-[#5547bd] shadow-sm" : "text-[#7d8792]"}`}>Monthly</button><button onClick={() => setTrend("week")} aria-pressed={trend === "week"} className={`rounded px-2.5 py-1.5 text-xs ${trend === "week" ? "bg-white font-semibold text-[#5547bd] shadow-sm" : "text-[#7d8792]"}`}>Weekly</button></div></div>{trendData.length ? <div className="mt-4 h-[260px]"><ResponsiveContainer width="100%" height="100%"><LineChart data={trendData} margin={{ top: 8, right: 8, left: -18, bottom: 0 }}><CartesianGrid vertical={false} stroke="#edf0f3"/><XAxis dataKey="period" tick={{ fontSize: 9, fill: "#87919c" }} axisLine={false} tickLine={false}/><YAxis yAxisId="revenue" tick={{ fontSize: 9, fill: "#87919c" }} axisLine={false} tickLine={false}/><YAxis yAxisId="orders" orientation="right" tick={{ fontSize: 9, fill: "#87919c" }} axisLine={false} tickLine={false}/><Tooltip/><Line yAxisId="revenue" type="monotone" dataKey="revenue" name="Revenue" stroke={accent} strokeWidth={2} dot={false} connectNulls={false}/><Line yAxisId="orders" type="monotone" dataKey="orders" name="Orders" stroke="#44ad91" strokeWidth={2} dot={false} connectNulls={false}/></LineChart></ResponsiveContainer></div> : <div className="grid h-[220px] place-items-center text-center text-sm text-[#8b95a0]">No valid dates or eligible revenue are available for this trend.<br/>Review date mapping and data quality.</div>}<p className="mt-2 text-[10px] text-[#9aa2ac]">Order trend counts unique order IDs. Dataset date range: {result.evidence.date_coverage.start ?? "unavailable"} to {result.evidence.date_coverage.end ?? "unavailable"}.</p></section>
            <section className={panelClass}><div className="flex items-start justify-between gap-2"><div><h2 className="font-semibold text-[#303945]">Top products</h2><p className="mt-1 text-xs text-[#89939f]">Choose a ranking based on the confirmed item-level source.</p></div><select aria-label="Product ranking" className="h-9 rounded-lg border border-[#e2e6ea] bg-white px-2 text-xs" value={productRanking} onChange={(event) => setProductRanking(event.target.value as "quantity" | "revenue")}><option value="quantity">By quantity</option><option value="revenue" disabled={!result.top_products.by_revenue.length}>By revenue</option></select></div>{productItems.length ? <div className="mt-4 h-[230px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={productItems} layout="vertical" margin={{ top: 2, right: 12, left: 8, bottom: 2 }}><CartesianGrid horizontal={false} stroke="#edf0f3"/><XAxis type="number" tick={{ fontSize: 9, fill: "#87919c" }} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="product" width={100} tick={{ fontSize: 9, fill: "#687482" }} axisLine={false} tickLine={false}/><Tooltip/><Bar dataKey={productRanking} name={productRanking === "quantity" ? "Quantity" : `Revenue${result.metrics.revenue.currency ? ` (${result.metrics.revenue.currency})` : ""}`} fill={accent} radius={[0,4,4,0]} maxBarSize={20}/></BarChart></ResponsiveContainer></div> : <p className="mt-5 text-sm text-[#8a94a0]">{productRanking === "revenue" ? result.top_products.reason ?? "Product revenue is not available for this source." : result.top_products.reason ?? "No product rows are available."}</p>}{result.top_products.reason && <p className="mt-2 text-xs text-[#a27040]">{result.top_products.reason}</p>}</section></div>
          <div className="mt-4 grid gap-4 xl:grid-cols-2"><section className={panelClass}><h2 className="font-semibold text-[#303945]">Category performance</h2>{result.categories.items.length ? <div className="mt-3 overflow-x-auto"><table className="w-full min-w-[440px] text-left text-xs"><thead className="text-[10px] uppercase tracking-wide text-[#929ba5]"><tr><th className="py-2">Category</th><th className="py-2">Distinct orders</th><th className="py-2">Order share</th><th className="py-2">Revenue</th></tr></thead><tbody>{result.categories.items.map((item) => <tr key={item.category} className="border-t border-[#eef0f3]"><td className="py-2.5 font-medium text-[#46515d]">{item.category}</td><td className="py-2.5 text-[#65717d]">{item.orders.toLocaleString()}</td><td className="py-2.5 text-[#65717d]">{item.order_contribution_pct === null ? "Unavailable" : `${item.order_contribution_pct}%`}</td><td className="py-2.5 text-[#65717d]">{item.revenue === null ? "Unavailable" : `${item.currency ? `${item.currency} ` : ""}${item.revenue.toLocaleString()}`}</td></tr>)}</tbody></table></div> : <p className="mt-3 text-sm text-[#89939f]">{result.categories.reason ?? "No category data is available."}</p>}{result.categories.reason && <p className="mt-2 text-xs text-[#a27040]">{result.categories.reason}</p>}</section>
            <section className={panelClass}><h2 className="font-semibold text-[#303945]">Data quality and assumptions</h2><div className="mt-3 grid gap-2 text-xs text-[#65717d] sm:grid-cols-2"><p>Source rows <b className="text-[#303945]">{result.source_rows.toLocaleString()}</b></p><p>Rows in selected range <b className="text-[#303945]">{result.selected_rows.toLocaleString()}</b></p><p>Invalid dates <b className="text-[#303945]">{String(result.data_quality.invalid_order_date_rows ?? "Not mapped")}</b></p><p>Exact duplicate rows <b className="text-[#303945]">{String(result.data_quality.exact_duplicate_rows_detected ?? 0)}</b></p><p>Currency mode <b className="text-[#303945]">{result.currency_status.replaceAll("_", " ")}</b></p><p>Comparison <b className="text-[#303945]">{result.period_comparison.status === "available" ? `${result.period_comparison.change_pct ?? "n/a"}% change` : result.period_comparison.reason ?? "Unavailable"}</b></p></div><details className="mt-3"><summary className="flex cursor-pointer list-none items-center gap-1 text-xs font-semibold text-[#5a4cc0]">Calculation evidence <ChevronDown size={13}/></summary><ul className="mt-2 list-disc space-y-1 pl-4 text-[11px] leading-5 text-[#78838f]">{result.evidence.assumptions.map((assumption) => <li key={assumption}>{assumption}</li>)}</ul><p className="mt-2 text-[10px] text-[#929ba5]">Dataset: {result.evidence.source_dataset}. Mapped columns: {Object.entries(result.evidence.source_columns).map(([role, column]) => `${role.replaceAll("_", " ")}: ${column}`).join(" · ") || "none"}</p><p className="mt-1 text-[10px] text-[#929ba5]">Revenue quality: {JSON.stringify(result.data_quality.revenue_quality ?? {})}</p></details></section></div>
          <p className="mt-4 rounded-lg bg-[#f7f7fb] p-3 text-xs leading-5 text-[#7a8490]">One saved CSV is analyzed at a time. Olist tables such as orders, order items, products, and payments are not automatically joined; multi-file relationship validation is not yet supported. Compare period changes as observed totals only; source completeness is not verified.</p>
        </>}
      </>}
    </>}
  </main>;
}
