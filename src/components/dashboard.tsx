"use client";

import { useCallback, useEffect, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { Activity, AlertCircle, ArrowDownToLine, BrainCircuit, ChevronLeft, ChevronRight, CloudUpload, Files, Grid2X2, Layers3, LoaderCircle, MessageCircle, Plus, Search, ShieldCheck, Sparkles, Table2, WandSparkles, X } from "lucide-react";
import { Area, AreaChart, Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { analyzeCsv } from "@/lib/api";
import { api, type Preferences } from "@/lib/auth";
import { downloadDatasetSource, reopenDataset } from "@/lib/datasets";
import MLStudio from "@/components/ml-studio";
import AIAnalystChat from "@/components/ai-analyst-chat";
import type { DatasetAnalysis } from "@/types/dataset";

const violet="#7566eb";
const chartColors=["#7668e9","#49bca5","#f3b95e","#ec8da7","#65a9dd"];

function fmt(n:number){return Intl.NumberFormat("en-US").format(n)}
function bytes(n:number){return n<1024*1024?`${(n/1024).toFixed(0)} KB`:`${(n/(1024*1024)).toFixed(1)} MB`}

function UploadPanel({onFile,loading,error}:{onFile:(f:File)=>void;loading:boolean;error:string|null}){const input=useRef<HTMLInputElement>(null);const [drag,setDrag]=useState(false);const validate=(file?:File)=>{if(!file)return;if(!file.name.toLowerCase().endsWith(".csv")){onFile(Object.assign(file,{name:file.name}));return}onFile(file)};const change=(e:ChangeEvent<HTMLInputElement>)=>{validate(e.target.files?.[0]);e.target.value=""};const drop=(e:DragEvent<HTMLDivElement>)=>{e.preventDefault();setDrag(false);validate(e.dataTransfer.files[0])};return <div onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={drop} className={`relative flex min-h-[216px] flex-col items-center justify-center overflow-hidden rounded-2xl border border-dashed bg-white px-6 py-6 text-center transition ${drag?"border-[#7566e9] bg-[#f7f5ff]":"border-[#d9dee5]"}`}><div className="pointer-events-none absolute inset-x-0 top-0 h-[3px] bg-gradient-to-r from-[#7a69ed] via-[#a58bfb] to-[#52c6ae] opacity-80"/><input ref={input} type="file" accept=".csv,text/csv" onChange={change} className="hidden"/><div className="mb-3 grid h-12 w-12 place-items-center rounded-[14px] bg-[#f0edff] text-[#7060df]">{loading?<LoaderCircle size={21} className="animate-spin"/>:<CloudUpload size={22}/>}</div><h3 className="display text-[15px] font-bold text-[#252e39]">{loading?"Analyzing your dataset…":"Drop your CSV file here"}</h3><p className="mt-1.5 text-[12px] text-[#8993a0]">or <button onClick={()=>input.current?.click()} className="font-semibold text-[#6e5fdd] hover:underline">browse files</button> from your computer</p><p className="mt-3 rounded-full bg-[#f5f6f8] px-3 py-1 text-[10px] font-medium text-[#909aa5]">CSV only · Max 25 MB</p>{error&&<div role="alert" className="mt-3 flex items-center gap-1.5 text-[11px] text-[#bd4f60]"><AlertCircle size={13}/>{error}</div>}</div>}

function MetricCard({label,value,sub,icon:Icon,accent}:{label:string;value:string;sub:string;icon:typeof Activity;accent:string}){return <div className="rounded-xl border border-[#e9ecf0] bg-white p-4"><div className="flex items-start justify-between"><span className="text-[11px] font-medium text-[#87919e]">{label}</span><span className={`grid h-7 w-7 place-items-center rounded-lg ${accent}`}><Icon size={14}/></span></div><p className="display mt-2 text-[24px] font-bold tracking-[-.8px] text-[#242d38]">{value}</p><p className="mt-1 text-[10px] text-[#a0a8b1]">{sub}</p></div>}

function OverviewPanel({data}:{data:DatasetAnalysis}){const nums=data.columns.filter(c=>c.numeric);const avg=nums.length?nums.reduce((n,c)=>n+(c.numeric?.mean??0),0)/nums.length:0;return <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><MetricCard label="Total records" value={fmt(data.row_count)} sub="Rows in current dataset" icon={Files} accent="bg-[#f0edff] text-[#7161e2]"/><MetricCard label="Features" value={fmt(data.column_count)} sub="Columns detected" icon={Layers3} accent="bg-[#e9f8f4] text-[#3eac94]"/><MetricCard label="Missing values" value={fmt(data.total_missing)} sub="Across all columns" icon={AlertCircle} accent="bg-[#fff5e8] text-[#dc9b3d]"/><MetricCard label="Numeric mean" value={avg?Intl.NumberFormat("en-US",{maximumFractionDigits:1}).format(avg):"—"} sub={nums.length?`Average of ${nums.length} numeric features`:"No numeric features found"} icon={Activity} accent="bg-[#eef5ff] text-[#5a8dcc]"/></div>}

function Charts({data}:{data:DatasetAnalysis}){const numeric=data.columns.filter(c=>c.numeric);const missing=data.columns.map(c=>({name:c.name,missing:c.missing,fill:data.row_count-c.missing}));const first=numeric[0];const distribution=first?.histogram??[];return <div className="grid gap-4 xl:grid-cols-[1.3fr_1fr]"><section className="rounded-xl border border-[#e9ecf0] bg-white p-5"><div className="mb-5 flex items-start justify-between"><div><h3 className="text-[13px] font-bold text-[#303946]">Feature distribution</h3><p className="mt-1 text-[11px] text-[#9aa2ac]">{first?`Distribution of ${first.name}`:"Missing values by feature"}</p></div><span className="rounded-md bg-[#f6f7f9] px-2 py-1 text-[10px] text-[#7f8995]">{first?.name??"All features"}</span></div><div className="h-[210px]">{first&&distribution.length?<ResponsiveContainer width="100%" height="100%"><BarChart data={distribution} margin={{top:5,right:4,left:-23,bottom:0}}><CartesianGrid vertical={false} stroke="#edf0f3"/><XAxis dataKey="bin" tick={{fontSize:9,fill:"#929ca8"}} axisLine={false} tickLine={false} interval="preserveStartEnd"/><YAxis tick={{fontSize:10,fill:"#929ca8"}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{border:"1px solid #e9ecf0",borderRadius:10,fontSize:11}}/><Bar dataKey="count" fill={violet} radius={[4,4,0,0]} maxBarSize={36}/></BarChart></ResponsiveContainer>:<ResponsiveContainer width="100%" height="100%"><BarChart data={missing} margin={{top:5,right:5,left:-20,bottom:0}}><CartesianGrid vertical={false} stroke="#edf0f3"/><XAxis dataKey="name" tick={{fontSize:9,fill:"#929ca8"}} axisLine={false} tickLine={false}/><YAxis tick={{fontSize:10,fill:"#929ca8"}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{border:"1px solid #e9ecf0",borderRadius:10,fontSize:11}}/><Bar dataKey="missing" fill="#efad5a" radius={[4,4,0,0]} maxBarSize={36}/></BarChart></ResponsiveContainer>}</div></section><section className="rounded-xl border border-[#e9ecf0] bg-white p-5"><div className="mb-5"><h3 className="text-[13px] font-bold text-[#303946]">Data completeness</h3><p className="mt-1 text-[11px] text-[#9aa2ac]">Non-missing values per feature</p></div><div className="h-[210px]"><ResponsiveContainer width="100%" height="100%"><AreaChart data={missing} margin={{top:5,right:5,left:-20,bottom:0}}><defs><linearGradient id="complete" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="#45b99d" stopOpacity={.19}/><stop offset="100%" stopColor="#45b99d" stopOpacity={0}/></linearGradient></defs><CartesianGrid vertical={false} stroke="#edf0f3"/><XAxis dataKey="name" tick={{fontSize:9,fill:"#929ca8"}} axisLine={false} tickLine={false}/><YAxis domain={[0,data.row_count||1]} tick={{fontSize:10,fill:"#929ca8"}} axisLine={false} tickLine={false}/><Tooltip contentStyle={{border:"1px solid #e9ecf0",borderRadius:10,fontSize:11}}/><Area type="monotone" dataKey="fill" stroke="#40b79b" strokeWidth={2} fill="url(#complete)"/></AreaChart></ResponsiveContainer></div></section></div>}

function TablePreview({data}:{data:DatasetAnalysis}){const [query,setQuery]=useState("");const [page,setPage]=useState(0);const [compact,setCompact]=useState(false);const rows=data.preview.filter(row=>!query||Object.values(row).some(value=>String(value??"").toLowerCase().includes(query.toLowerCase())));const pageSize=5;const shown=rows.slice(page*pageSize,(page+1)*pageSize);useEffect(()=>{setPage(0)},[query]);useEffect(()=>{api<{preferences:Preferences}>("/auth/preferences").then(({preferences})=>setCompact(preferences.compact_tables)).catch(()=>undefined)},[]);return <section className="overflow-hidden rounded-xl border border-[#e9ecf0] bg-white"><div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#eef0f3] px-5 py-4"><div><h3 className="text-[13px] font-bold text-[#303946]">Data preview</h3><p className="mt-1 text-[11px] text-[#9aa2ac]">Showing {Math.min(pageSize,rows.length)} of {data.row_count} rows</p></div><div className="flex gap-2"><label className="flex h-8 items-center gap-2 rounded-lg border border-[#e8ebef] px-2.5"><Search size={13} className="text-[#a0a8b1]"/><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search rows…" className="w-28 bg-transparent text-[11px] outline-none placeholder:text-[#a0a8b1] sm:w-36"/></label></div></div><div className="scrollbar-hide overflow-x-auto"><table className="w-full min-w-[560px] text-left"><thead className="bg-[#fafbfc]"><tr>{data.columns.map(c=><th key={c.name} className="px-4 py-3 text-[10px] font-semibold uppercase tracking-[.6px] text-[#858f9c] first:pl-5"><span className="mr-1.5 text-[#a69aeb]">{c.kind==="numeric"?"#":"A"}</span>{c.name}</th>)}</tr></thead><tbody>{shown.map((row,index)=><tr key={index} className="border-t border-[#f0f2f4] hover:bg-[#fafbfc]">{data.columns.map(c=><td key={c.name} className={`max-w-[200px] truncate px-4 ${compact?"py-1.5":"py-3"} text-[11px] text-[#55616f] first:pl-5`}>{row[c.name]===null?<span className="rounded bg-[#fff4e9] px-1.5 py-0.5 text-[10px] text-[#cc9150]">null</span>:String(row[c.name])}</td>)}</tr>)}</tbody></table>{!shown.length&&<p className="p-8 text-center text-[12px] text-[#99a1ab]">No preview rows match your search.</p>}</div><div className="flex items-center justify-between border-t border-[#eef0f3] px-5 py-3"><span className="text-[10px] text-[#9aa2ac]">{data.duplicate_rows?`${fmt(data.duplicate_rows)} duplicate rows detected · `:""}Showing first {data.preview.length} rows</span><div className="flex items-center gap-1"><button onClick={()=>setPage(Math.max(0,page-1))} className="grid h-7 w-7 place-items-center rounded border border-[#e8ebef] text-[#76818e] disabled:opacity-40" disabled={page===0}><ChevronLeft size={14}/></button><span className="px-2 text-[10px] text-[#79838f]">{page+1} / {Math.max(1,Math.ceil(rows.length/pageSize))}</span><button onClick={()=>setPage(Math.min(Math.ceil(rows.length/pageSize)-1,page+1))} className="grid h-7 w-7 place-items-center rounded border border-[#e8ebef] text-[#76818e] disabled:opacity-40" disabled={page>=Math.ceil(rows.length/pageSize)-1}><ChevronRight size={14}/></button></div></div></section>}

function ColumnPanel({data}:{data:DatasetAnalysis}){return <section className="overflow-hidden rounded-xl border border-[#e9ecf0] bg-white"><div className="flex items-center justify-between border-b border-[#eef0f3] px-5 py-4"><div><h3 className="text-[13px] font-bold text-[#303946]">Column insights</h3><p className="mt-1 text-[11px] text-[#9aa2ac]">Types and quick summaries for each feature</p></div></div><div className="scrollbar-hide overflow-x-auto"><table className="w-full min-w-[650px] text-left"><thead className="bg-[#fafbfc]"><tr>{["Feature","Type","Missing","Unique","Mean / mode","Range / top value"].map(x=><th key={x} className="px-4 py-3 text-[10px] font-semibold uppercase tracking-[.55px] text-[#88929e] first:pl-5">{x}</th>)}</tr></thead><tbody>{data.columns.map((c,index)=><tr key={c.name} className="border-t border-[#f0f2f4] hover:bg-[#fafbfc]"><td className="px-4 py-3.5 text-[11px] font-semibold text-[#424c59] first:pl-5"><span className={`mr-2 inline-block h-2 w-2 rounded-full`} style={{background:chartColors[index%chartColors.length]}}/>{c.name}</td><td className="px-4 py-3.5"><span className={`rounded-md px-2 py-1 text-[9px] font-semibold ${c.kind==="numeric"?"bg-[#f0edff] text-[#6c5bde]":"bg-[#eef7f4] text-[#419b84]"}`}>{c.kind}</span></td><td className="px-4 py-3.5 text-[11px] text-[#55616f]">{fmt(c.missing)} <span className="text-[10px] text-[#9aa2ac]">({c.missing_pct}%)</span></td><td className="px-4 py-3.5 text-[11px] text-[#55616f]">{fmt(c.unique)}</td><td className="px-4 py-3.5 text-[11px] text-[#55616f]">{c.numeric?.mean===null||c.numeric?.mean===undefined?c.top_values[0]?.value??"—":c.numeric.mean.toLocaleString(undefined,{maximumFractionDigits:2})}</td><td className="px-4 py-3.5 text-[11px] text-[#55616f]">{c.numeric?`${c.numeric.min} – ${c.numeric.max}`:c.top_values[0]?`${c.top_values[0].value} (${c.top_values[0].count})`:"—"}</td></tr>)}</tbody></table></div></section>}

function CleaningDialog({data,onClose,onApply,loading}:{data:DatasetAnalysis;onClose:()=>void;onApply:(x:{drop_duplicates:boolean;fill_numeric:boolean;fill_categorical:boolean})=>void;loading:boolean}){const [drop,setDrop]=useState(false),[numeric,setNumeric]=useState(false),[categorical,setCategorical]=useState(false);const Toggle=({value,set}:{value:boolean;set:(v:boolean)=>void})=><button onClick={()=>set(!value)} className={`relative h-5 w-9 rounded-full transition ${value?"bg-[#7566e9]":"bg-[#d9dee5]"}`}><span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-all ${value?"left-[18px]":"left-0.5"}`}/></button>;return <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#1c2532]/35 p-4 backdrop-blur-[2px]" onClick={onClose}><div className="w-full max-w-[440px] rounded-2xl border border-[#e7e9ed] bg-white p-6 shadow-xl" onClick={e=>e.stopPropagation()}><div className="flex items-start justify-between"><div><h3 className="display text-[17px] font-bold text-[#26303b]">Clean your dataset</h3><p className="mt-1 text-[12px] text-[#8b95a1]">Preview changes derived from {data.filename}</p></div><button onClick={onClose} className="text-[#929ba6]"><X size={18}/></button></div><div className="mt-5 space-y-3">{[{title:"Remove duplicate rows",desc:`${fmt(data.duplicate_rows)} exact duplicates found`,value:drop,set:setDrop},{title:"Fill numeric missing values",desc:"Replace blanks with each column’s median",value:numeric,set:setNumeric},{title:"Fill categorical missing values",desc:"Replace blanks with each column’s most common value",value:categorical,set:setCategorical}].map(row=><div key={row.title} className="flex items-center justify-between rounded-xl border border-[#eceef1] p-3.5"><div><p className="text-[12px] font-semibold text-[#384351]">{row.title}</p><p className="mt-1 text-[10px] text-[#919aa5]">{row.desc}</p></div><Toggle value={row.value} set={row.set}/></div>)}</div><div className="mt-4 flex items-center gap-2 rounded-lg bg-[#f5f4ff] p-3 text-[10px] leading-relaxed text-[#7669bf]"><ShieldCheck size={15} className="shrink-0"/>Changes create an analysis preview. Your original file is never altered.</div><div className="mt-5 flex justify-end gap-2"><button onClick={onClose} className="rounded-lg border border-[#e6e9ed] px-4 py-2 text-[11px] font-semibold text-[#6f7a87]">Cancel</button><button disabled={loading} onClick={()=>onApply({drop_duplicates:drop,fill_numeric:numeric,fill_categorical:categorical})} className="flex items-center gap-2 rounded-lg bg-[#6e60df] px-4 py-2 text-[11px] font-semibold text-white disabled:opacity-60">{loading&&<LoaderCircle size={13} className="animate-spin"/>}Apply to preview</button></div></div></div>}


const tabViews: Record<string, string> = { Overview: "overview", Columns: "columns", "Data preview": "preview", "ML Studio": "ml", "AI Chat": "ai" };
const viewTabs: Record<string, string> = Object.fromEntries(Object.entries(tabViews).map(([tab, view]) => [view, tab]));

export default function Dashboard({ datasetId, initialTab }: { datasetId?: string; initialTab?: string }) {
  const router = useRouter();
  const [data, setData] = useState<DatasetAnalysis | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [dialog, setDialog] = useState(false);
  const [cleaning, setCleaning] = useState(false);
  const [tab, setTab] = useState(initialTab ? viewTabs[initialTab] ?? "Overview" : "Overview");
  const [storageStatus, setStorageStatus] = useState<{ database: string; dataset_storage: string } | null>(null);

  useEffect(() => {
    let active = true;
    api<{ database: string; dataset_storage: string }>("/runtime/status").then((value) => { if (active) setStorageStatus(value); }).catch(() => undefined);
    return () => { active = false; };
  }, []);

  useEffect(() => {
    if (!datasetId) return;
    let active = true;
    setLoading(true);
    setError(null);
    reopenDataset(datasetId)
      .then(async (analysis) => {
        const source = await downloadDatasetSource(datasetId, analysis.filename);
        if (!active) return;
        setData(analysis);
        setFile(source);
      })
      .catch((cause: unknown) => {
        if (active) setError(cause instanceof Error ? cause.message : "Could not reopen this dataset.");
      })
      .finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [datasetId]);

  const selectTab = useCallback((nextTab: string) => {
    setTab(nextTab);
    if (datasetId) router.replace(`/datasets/${encodeURIComponent(datasetId)}?view=${tabViews[nextTab]}`, { scroll: false });
  }, [datasetId, router]);


  const perform = useCallback(async (selected: File, options?: { drop_duplicates: boolean; fill_numeric: boolean; fill_categorical: boolean }) => {
    setData(null);
    setFile(null);
    setError(null);
    if (!selected.name.toLowerCase().endsWith(".csv")) {
      setError("Please choose a CSV file.");
      return;
    }
    if (selected.size > 25 * 1024 * 1024) {
      setError("File exceeds the 25 MB upload limit.");
      return;
    }
    setLoading(true);
    setCleaning(!!options);
    setError(null);
    try {
      const result = await analyzeCsv(selected, options);
      setFile(selected);
      setData(result);
      selectTab("Overview");
      if (!options && result.dataset_id) router.push(`/datasets/${encodeURIComponent(result.dataset_id)}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not analyze CSV.");
    } finally {
      setLoading(false);
      setCleaning(false);
    }
  }, [router, selectTab]);

  const exportCsv = () => {
    if (!data) return;
    const header = data.columns.map((column) => column.name);
    const escapeCell = (value: unknown) => {
      let text = String(value ?? "");
      if (/^[\s]*[=+@\-]/.test(text)) text = `'${text}`;
      return `"${text.replaceAll('"', '""')}"`;
    };
    const lines = [header, ...data.preview.map((row) => header.map((name) => row[name] ?? ""))]
      .map((row) => row.map(escapeCell).join(","));
    const link = document.createElement("a");
    const url = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    link.href = url;
    link.download = `${data.filename.replace(/\.csv$/i, "")}-preview.csv`;
    link.click();
    URL.revokeObjectURL(url);
  };

  return <><main className="mx-auto max-w-[1600px] px-5 py-7 sm:px-7 lg:px-9">
    <div className="mb-6 flex flex-wrap items-end justify-between gap-4"><div><div className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[1px] text-[#8e98a5]"><span className="h-1.5 w-1.5 rounded-full bg-[#47b99d]"/>Data workspace</div><h1 className="display text-[25px] font-extrabold tracking-[-1px] text-[#252f3a] sm:text-[28px]">Your data workspace <span className="text-[#e8ad59]">✳</span></h1><p className="mt-1.5 text-[12px] text-[#88929e]">Upload a CSV to explore its real statistics and prepare a clean preview.</p></div>{datasetId ? <Link href="/new-analysis" className="flex h-9 items-center gap-2 rounded-lg bg-[#6f60df] px-3.5 text-[11px] font-semibold text-white"><Plus size={15}/>New analysis</Link> : <button onClick={()=>document.getElementById("upload")?.scrollIntoView({behavior:"smooth",block:"center"})} className="flex h-9 items-center gap-2 rounded-lg bg-[#6f60df] px-3.5 text-[11px] font-semibold text-white"><Plus size={15}/>Upload a CSV</button>}</div>
    {datasetId ? (!data && loading ? <div role="status" className="rounded-xl border border-[#e8ebef] bg-white p-6 text-sm text-[#85909c]">Reopening saved CSV and rebuilding its preview…</div> : !data && error ? <div role="alert" className="rounded-xl border border-[#f1d7d5] bg-white p-5 text-sm text-[#a04750]">{error}<div className="mt-3"><Link href="/new-analysis" className="font-semibold text-[#6f60df]">Choose a CSV to reupload</Link></div></div> : null) : <div className="grid gap-5 xl:grid-cols-[1.45fr_1fr]"><div id="upload"><UploadPanel onFile={(selected)=>perform(selected)} loading={loading&&!cleaning} error={error}/></div><section className="rounded-2xl border border-[#ebe8ff] bg-gradient-to-br from-[#f6f4ff] via-white to-[#f5fbfa] p-5"><span className="inline-flex items-center gap-1.5 rounded-full bg-white px-2.5 py-1 text-[9px] font-bold uppercase tracking-[.7px] text-[#7163d6] shadow-sm"><Sparkles size={11}/>Quick start</span><h2 className="display mt-3 text-[16px] font-bold text-[#31364a]">Make sense of your data</h2><p className="mt-1 max-w-[380px] text-[11px] leading-relaxed text-[#85889a]">Upload a CSV and get a private snapshot with computed summaries, missing values, distributions, and a row preview.</p><div className="mt-4 grid grid-cols-3 gap-2">{[["01","Upload","Your CSV file"],["02","Explore","Stats & charts"],["03","Prepare","Clean safely"]].map(([number,title,description])=><div key={number} className="rounded-lg border border-white bg-white/70 p-2.5"><span className="text-[9px] font-bold text-[#8273dc]">{number}</span><p className="mt-1 text-[10px] font-bold text-[#4b5360]">{title}</p><p className="mt-0.5 text-[9px] text-[#a0a6af]">{description}</p></div>)}</div></section></div>}
    {data&&file&&<section id="results" className="mt-9 scroll-mt-6"><div className="mb-4 flex flex-wrap items-center justify-between gap-3"><div><div className="flex items-center gap-2"><h2 className="display text-[17px] font-bold text-[#2e3844]">Dataset overview</h2><span className="rounded-md bg-[#f0edff] px-2 py-1 text-[9px] font-semibold text-[#6b5dde]">{data.filename}</span>{cleaning&&<span className="rounded-md bg-[#eaf8f4] px-2 py-1 text-[9px] font-semibold text-[#379b82]">Cleaned preview</span>}</div><p className="mt-1 text-[11px] text-[#929ca7]">Computed from uploaded data · {bytes(data.size_bytes)} · original file preserved</p></div><div className="flex gap-2"><button onClick={()=>setDialog(true)} className="flex h-8 items-center gap-1.5 rounded-lg border border-[#e5e8ec] bg-white px-3 text-[10px] font-semibold text-[#687482]"><WandSparkles size={13}/>Clean data</button><button onClick={exportCsv} className="grid h-8 w-8 place-items-center rounded-lg border border-[#e5e8ec] bg-white text-[#687482]" title="Export preview"><ArrowDownToLine size={14}/></button></div></div><div className="mb-4 flex gap-1 overflow-x-auto border-b border-[#e8ebef]">{[{name:"Overview",icon:Grid2X2},{name:"Columns",icon:Layers3},{name:"Data preview",icon:Table2},{name:"ML Studio",icon:BrainCircuit},{name:"AI Chat",icon:MessageCircle}].map(({name,icon:Icon})=><button key={name} onClick={()=>selectTab(name)} className={`flex shrink-0 items-center gap-1.5 border-b-2 px-3 py-2.5 text-[11px] font-semibold ${tab===name?"border-[#7667e4] text-[#685bd2]":"border-transparent text-[#929ba6] hover:text-[#596574]"}`}><Icon size={13}/>{name}</button>)}</div>{tab==="Overview"?<><OverviewPanel data={data}/><div className="mt-4"><Charts data={data}/></div><div className="mt-4"><TablePreview data={data}/></div><div className="mt-4"><ColumnPanel data={data}/></div></>:tab==="Columns"?<ColumnPanel data={data}/>:tab==="Data preview"?<TablePreview data={data}/>:tab==="ML Studio"?<MLStudio key={`${data.filename}:${data.size_bytes}:${file.lastModified}`} file={file} data={data}/>:<AIAnalystChat key={datasetId} datasetId={datasetId!} filename={data.filename}/>}</section>}
    <footer className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t border-[#e9ecf0] pt-4 text-[9px] text-[#8e98a5]"><span>NEXORA AI · Private analytics workspace</span><span>{storageStatus ? `Storage: ${storageStatus.database} · ${storageStatus.dataset_storage}` : "Storage status unavailable"}</span></footer>
  </main>{dialog&&data&&file&&<CleaningDialog data={data} loading={loading} onClose={()=>setDialog(false)} onApply={(options)=>{setDialog(false);void perform(file,options)}}/>}</>;
}










