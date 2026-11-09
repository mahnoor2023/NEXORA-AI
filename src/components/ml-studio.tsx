"use client";

import { useState } from "react";
import { AlertCircle, BrainCircuit, LoaderCircle, ShieldCheck, Sparkles } from "lucide-react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { trainCsv } from "@/lib/api";
import type { DatasetAnalysis, MLTask, MLTrainingResult } from "@/types/dataset";

const modelColors = ["#7566e9", "#44b59b"];
const metricLabels: Record<string, string> = {
  accuracy: "Accuracy",
  balanced_accuracy: "Balanced accuracy",
  precision_weighted: "Weighted precision",
  recall_weighted: "Weighted recall",
  f1_weighted: "Weighted F1",
  mae: "MAE",
  rmse: "RMSE",
  r2: "R²",
};

function Metric({ label, value }: { label: string; value: number | string }) {
  const formatted = typeof value === "number" ? value.toLocaleString(undefined, { maximumFractionDigits: 3 }) : value;
  return <div className="rounded-lg bg-[#f8f9fb] p-3"><p className="text-[9px] font-medium text-[#8b95a1]">{label}</p><p className="display mt-1 text-[16px] font-bold text-[#34404d]">{formatted}</p></div>;
}

export default function MLStudio({ file, data }: { file: File; data: DatasetAnalysis }) {
  const [target, setTarget] = useState("");
  const [task, setTask] = useState<MLTask>("auto");
  const [training, setTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<MLTrainingResult | null>(null);

  const runTraining = async () => {
    if (!target) {
      setError("Choose the column you want the model to predict.");
      return;
    }
    setTraining(true);
    setError(null);
    setResult(null);
    try {
      setResult(await trainCsv(file, target, task));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not train models.");
    } finally {
      setTraining(false);
    }
  };

  const best = result?.models.find((model) => model.name === result.best_model);
  const comparison = result?.models.map((model) => ({
    name: model.name,
    score: model.metrics[result.primary_metric],
  })) ?? [];

  return <section className="space-y-4">
    <div className="rounded-xl border border-[#e9ecf0] bg-white p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div><div className="flex items-center gap-2"><span className="grid h-8 w-8 place-items-center rounded-lg bg-[#f0edff] text-[#6e5fdd]"><BrainCircuit size={17}/></span><h3 className="display text-[15px] font-bold text-[#303946]">Train baseline models</h3></div><p className="mt-2 max-w-[620px] text-[11px] leading-relaxed text-[#929ba6]">Choose a target to compare two lightweight supervised models on a fixed-seed holdout split. Preprocessing is fitted on training rows only.</p></div>
        <span className="rounded-full bg-[#f5f6f8] px-2.5 py-1 text-[9px] text-[#818b97]">Source: {data.filename}</span>
      </div>
      <div className="mt-5 grid gap-3 sm:grid-cols-[1fr_210px_auto] sm:items-end">
        <label className="block"><span className="mb-1.5 block text-[10px] font-semibold text-[#6e7986]">Target column</span><select value={target} onChange={(event) => setTarget(event.target.value)} className="h-9 w-full rounded-lg border border-[#e3e7eb] bg-white px-3 text-[11px] text-[#485361] outline-none focus:border-[#9488e8]"><option value="">Select a column…</option>{data.columns.map((column) => <option key={column.name} value={column.name}>{column.name} · {column.kind}</option>)}</select></label>
        <label className="block"><span className="mb-1.5 block text-[10px] font-semibold text-[#6e7986]">Problem type</span><select value={task} onChange={(event) => setTask(event.target.value as MLTask)} className="h-9 w-full rounded-lg border border-[#e3e7eb] bg-white px-3 text-[11px] text-[#485361] outline-none focus:border-[#9488e8]"><option value="auto">Auto-detect</option><option value="classification">Classification</option><option value="regression">Regression</option></select></label>
        <button onClick={runTraining} disabled={training || !target} className="flex h-9 items-center justify-center gap-2 rounded-lg bg-[#6f60df] px-4 text-[11px] font-semibold text-white transition hover:bg-[#5f51ce] disabled:cursor-not-allowed disabled:opacity-50">{training ? <><LoaderCircle size={14} className="animate-spin"/>Training…</> : <><BrainCircuit size={14}/>Train models</>}</button>
      </div>
      <div className="mt-3 flex items-start gap-1.5 text-[9px] leading-relaxed text-[#909aa5]"><ShieldCheck size={12} className="mt-px shrink-0 text-[#54a994]"/>Training runs on the configured NEXORA API using your uploaded CSV. Classification supports up to 50 classes and needs at least two examples per class; continuous numeric targets are usually better suited to regression. Exact target copies are rejected; hidden domain or time leakage still needs review.</div>
      {error && <div role="alert" className="mt-4 flex items-start gap-2 rounded-lg border border-[#f1d9dc] bg-[#fff8f8] p-3 text-[11px] text-[#b34e5d]"><AlertCircle size={14} className="mt-px shrink-0"/>{error}</div>}
    </div>

    {result && best && <>
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <Metric label="Task" value={result.task === "classification" ? "Classification" : "Regression"}/>
        <Metric label="Training rows" value={result.train_rows}/>
        <Metric label="Held-out rows" value={result.test_rows}/>
        <Metric label="Features" value={result.feature_count}/>
      </div>
      <div className="grid gap-4 xl:grid-cols-[1fr_1.2fr]">
        <section className="rounded-xl border border-[#e9ecf0] bg-white p-5">
          <div className="flex flex-wrap items-start justify-between gap-2"><div><h3 className="text-[13px] font-bold text-[#303946]">Model comparison</h3><p className="mt-1 text-[10px] text-[#99a2ac]">Higher {metricLabels[result.primary_metric]} ranks first</p></div><span className="rounded-md bg-[#f0edff] px-2 py-1 text-[9px] font-semibold text-[#6c5bde]">Best: {result.best_model}</span></div>
          <div className="mt-3 h-[190px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={comparison} margin={{ top: 8, right: 6, left: -18, bottom: 0 }}><CartesianGrid vertical={false} stroke="#edf0f3"/><XAxis dataKey="name" tick={{ fontSize: 9, fill: "#7e8996" }} axisLine={false} tickLine={false}/><YAxis tick={{ fontSize: 9, fill: "#929ca8" }} axisLine={false} tickLine={false}/><Tooltip contentStyle={{ border: "1px solid #e9ecf0", borderRadius: 9, fontSize: 10 }}/><Bar dataKey="score" name={metricLabels[result.primary_metric]} radius={[4,4,0,0]} maxBarSize={42}>{comparison.map((entry, index) => <Cell key={entry.name} fill={modelColors[index % modelColors.length]}/>)}</Bar></BarChart></ResponsiveContainer></div>
          <div className="mt-3 grid gap-2 sm:grid-cols-2">{result.models.map((model) => <div key={model.name} className="rounded-lg border border-[#eef0f3] p-3"><p className="mb-2 text-[10px] font-bold text-[#485361]">{model.name}</p><div className="grid grid-cols-2 gap-2">{Object.entries(model.metrics).map(([name, value]) => <Metric key={name} label={metricLabels[name] ?? name} value={value}/>)}</div></div>)}</div>
        </section>
        <section className="rounded-xl border border-[#e9ecf0] bg-white p-5">
          <div className="flex items-start justify-between"><div><h3 className="text-[13px] font-bold text-[#303946]">{result.insights.provider === "llm" ? "LLM-assisted insights" : "Rule-based ML insights"}</h3><p className="mt-1 text-[10px] text-[#99a2ac]">Grounded in this run’s computed metrics</p></div><span className="flex items-center gap-1 rounded-full bg-[#f4f2ff] px-2 py-1 text-[9px] font-semibold text-[#7163d6]"><Sparkles size={11}/>{result.insights.label}</span></div>
          <h4 className="mt-4 text-[12px] font-bold text-[#3b4653]">{result.insights.title}</h4><p className="mt-2 text-[11px] leading-relaxed text-[#65717e]">{result.insights.summary}</p>
          {!!result.insights.fallback_reason && <p className="mt-2 text-[10px] text-[#a07b3b]">{result.insights.fallback_reason}</p>}
          {!!result.insights.bullets.length && <ul className="mt-3 space-y-2">{result.insights.bullets.map((bullet) => <li key={bullet} className="flex gap-2 text-[10px] leading-relaxed text-[#737e8a]"><span className="mt-1 h-1 w-1 shrink-0 rounded-full bg-[#7667e4]"/>{bullet}</li>)}</ul>}
          <div className="mt-4 border-t border-[#eef0f3] pt-3"><p className="text-[9px] font-bold uppercase tracking-wide text-[#929ba6]">Limitations</p><ul className="mt-2 space-y-1.5">{result.insights.limitations.map((item) => <li key={item} className="text-[9px] leading-relaxed text-[#89939f]">• {item}</li>)}</ul></div>
          {result.insights.provider === "llm" && <p className="mt-3 rounded-md bg-[#f8f7ff] p-2 text-[9px] leading-relaxed text-[#7569b7]">Only aggregate metrics and top feature-importance values were sent to the server-configured LLM provider. The API key stays on the server.</p>}
        </section>
      </div>
      <div className="grid gap-4 xl:grid-cols-2">
        <section className="rounded-xl border border-[#e9ecf0] bg-white p-5"><h3 className="text-[13px] font-bold text-[#303946]">Feature importance</h3><p className="mt-1 text-[10px] text-[#99a2ac]">Predictive signal from {best.name}; not causal</p>{best.feature_importance.length ? <div className="mt-3 h-[220px]"><ResponsiveContainer width="100%" height="100%"><BarChart data={best.feature_importance} layout="vertical" margin={{ top: 2, right: 12, left: 18, bottom: 2 }}><CartesianGrid horizontal={false} stroke="#edf0f3"/><XAxis type="number" tick={{ fontSize: 9, fill: "#929ca8" }} axisLine={false} tickLine={false}/><YAxis type="category" dataKey="feature" width={100} tick={{ fontSize: 9, fill: "#737e8a" }} axisLine={false} tickLine={false}/><Tooltip contentStyle={{ border: "1px solid #e9ecf0", borderRadius: 9, fontSize: 10 }}/><Bar dataKey="importance" fill="#7566e9" radius={[0,4,4,0]} maxBarSize={18}/></BarChart></ResponsiveContainer></div> : <p className="mt-4 text-[11px] text-[#8b95a1]">This estimator did not expose supported feature-importance values.</p>}</section>
        <section className="overflow-hidden rounded-xl border border-[#e9ecf0] bg-white"><div className="border-b border-[#eef0f3] px-5 py-4"><h3 className="text-[13px] font-bold text-[#303946]">Held-out predictions</h3><p className="mt-1 text-[10px] text-[#99a2ac]">First {best.predictions.length} of {result.test_rows} test rows · {result.target}</p></div><div className="scrollbar-hide max-h-[260px] overflow-auto"><table className="w-full text-left"><thead className="sticky top-0 bg-[#fafbfc]"><tr><th className="px-4 py-2.5 text-[9px] uppercase text-[#89939f]">Row</th><th className="px-4 py-2.5 text-[9px] uppercase text-[#89939f]">Actual</th><th className="px-4 py-2.5 text-[9px] uppercase text-[#89939f]">Predicted</th><th className="px-4 py-2.5 text-[9px] uppercase text-[#89939f]">Result</th></tr></thead><tbody>{best.predictions.map((prediction, index) => { const equal = String(prediction.actual) === String(prediction.predicted); return <tr key={index} className="border-t border-[#f0f2f4]"><td className="px-4 py-2.5 text-[10px] text-[#87919d]">{index + 1}</td><td className="px-4 py-2.5 text-[10px] text-[#53606e]">{String(prediction.actual)}</td><td className="px-4 py-2.5 text-[10px] text-[#53606e]">{String(prediction.predicted)}</td><td className="px-4 py-2.5"><span className={`rounded px-1.5 py-0.5 text-[9px] ${equal ? "bg-[#eaf8f4] text-[#3b987f]" : "bg-[#fff3e9] text-[#bd8544]"}`}>{result.task === "regression" ? "scored" : equal ? "match" : "miss"}</span></td></tr>; })}</tbody></table></div></section>
      </div>
      {result.task === "classification" && best.confusion_matrix && <section className="rounded-xl border border-[#e9ecf0] bg-white p-5"><h3 className="text-[13px] font-bold text-[#303946]">Confusion matrix · {best.name}</h3><p className="mt-1 text-[10px] text-[#99a2ac]">Rows are actual labels; columns are predicted labels.</p><div className="mt-3 overflow-x-auto"><table className="min-w-[300px] border-collapse text-center"><thead><tr><th className="p-2 text-[9px] text-[#87919d]">Actual ↓ / Predicted →</th>{best.class_labels.map((label) => <th key={String(label)} className="p-2 text-[9px] text-[#687482]">{String(label)}</th>)}</tr></thead><tbody>{best.confusion_matrix.map((row, index) => <tr key={index}><th className="p-2 text-left text-[9px] text-[#687482]">{String(best.class_labels[index])}</th>{row.map((count, column) => <td key={column} className="border border-white p-2 text-[10px] text-[#45515f]" style={{ backgroundColor: `rgba(117, 102, 233, ${0.06 + 0.5 * count / Math.max(1, ...row)})` }}>{count}</td>)}</tr>)}</tbody></table></div></section>}
      <p className="text-[9px] text-[#9aa3ad]">Seed {result.random_seed} · {result.train_rows.toLocaleString()} train / {result.test_rows.toLocaleString()} test · completed in {result.training_seconds}s</p>
    </>}
  </section>;
}
