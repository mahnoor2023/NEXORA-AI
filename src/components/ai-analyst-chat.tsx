"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { AlertCircle, Bot, Check, LoaderCircle, Send, ShieldCheck, Sparkles } from "lucide-react";
import { AgentRequestError, askAnalyst, getAnalystStatus } from "@/lib/api";
import type { AgentChatResult, AnalystStatus } from "@/types/dataset";

type ChatMessage = { role: "user"; text: string } | { role: "assistant"; result: AgentChatResult };

export default function AIAnalystChat({ datasetId, filename }: { datasetId: string; filename: string }) {
  const [status, setStatus] = useState<AnalystStatus | null>(null);
  const [statusFailed, setStatusFailed] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let active = true;
    getAnalystStatus().then((value) => { if (active) setStatus(value); }).catch(() => { if (active) setStatusFailed(true); });
    return () => { active = false; };
  }, []);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }, [messages, loading]);

  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const prompt = question.trim();
    if (!prompt || loading) return;
    setQuestion("");
    setError(null);
    setMessages((current) => [...current, { role: "user", text: prompt }]);
    setLoading(true);
    try {
      const result = await askAnalyst(datasetId, prompt);
      setMessages((current) => [...current, { role: "assistant", result }]);
    } catch (cause) {
      setError(cause instanceof AgentRequestError ? cause.message : cause instanceof Error ? cause.message : "Pandas analysis could not be completed.");
    } finally {
      setLoading(false);
    }
  };

  return <section className="overflow-hidden rounded-xl border border-[#e9ecf0] bg-white">
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#eef0f3] px-5 py-4">
      <div className="flex items-center gap-2.5"><span className="grid h-9 w-9 place-items-center rounded-[10px] bg-[#f0edff] text-[#6f60df]"><Bot size={18}/></span><div><h3 className="text-[13px] font-bold text-[#303946]">Pandas Data Analysis Assistant</h3><p className="mt-0.5 text-[10px] text-[#99a2ac]">Deterministic analysis · {filename}</p></div></div>
      <span className="flex items-center gap-1.5 rounded-full bg-[#eaf8f4] px-2.5 py-1 text-[9px] font-semibold text-[#398f7b]"><Check size={11}/>Ready · no LLM required</span>
    </div>

    <div aria-live="polite" className="scrollbar-hide flex min-h-[280px] max-h-[520px] flex-col gap-3 overflow-y-auto bg-[#fafbfc] p-4 sm:p-5">
      {!messages.length && <div className="m-auto max-w-[440px] py-8 text-center"><span className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-white text-[#7869e6] shadow-sm"><Sparkles size={19}/></span><h4 className="mt-3 text-[12px] font-bold text-[#485361]">Ask a question about your data</h4><p className="mt-1.5 text-[10px] leading-relaxed text-[#929ca7]">Try “How many rows?”, “Which columns have missing values?”, “What is the average revenue?”, “Top 5 seller states by number of sellers”, or “Which seller states have fewer than 5 sellers?”. Unsupported or ambiguous requests are identified clearly.</p></div>}
      {messages.map((message, index) => message.role === "user"
        ? <div key={index} className="ml-auto max-w-[85%] rounded-xl rounded-br-sm bg-[#6f60df] px-3.5 py-2.5 text-[11px] leading-relaxed text-white">{message.text}</div>
        : <article key={index} className="max-w-[92%] rounded-xl rounded-bl-sm border border-[#e9ecf0] bg-white p-3.5 shadow-[0_2px_9px_#26303b06]"><div className="mb-2 flex items-center gap-1.5 text-[9px] font-semibold text-[#7062d8]"><Bot size={12}/>{message.result.status === "unsupported" ? "Request needs clarification" : "Computed with Pandas"}</div><p className="whitespace-pre-wrap text-[11px] leading-relaxed text-[#4e5a67]">{message.result.answer}</p>{message.result.citations.length > 0 && <details className="mt-2.5"><summary className="cursor-pointer text-[9px] font-semibold text-[#7c8793]">Computed evidence</summary><ul className="mt-2 space-y-1.5">{message.result.citations.map((citation, citeIndex) => <li key={`${citation.tool}-${citeIndex}`} className="rounded-md bg-[#f8f9fb] p-2 text-[9px] leading-relaxed text-[#6f7b88]"><span className="font-semibold text-[#7163d6]">{citation.tool}: </span>{citation.quote}</li>)}</ul></details>}<p className="mt-2 flex items-start gap-1.5 text-[8px] leading-relaxed text-[#a0a8b1]"><ShieldCheck size={10} className="mt-px shrink-0 text-[#54a994]"/>{message.result.privacy}</p></article>)}
      {loading && <div role="status" className="flex w-fit items-center gap-2 rounded-lg border border-[#e9ecf0] bg-white px-3 py-2 text-[10px] text-[#7e8995]"><LoaderCircle size={13} className="animate-spin text-[#7667e4]"/>Computing results from the saved dataset…</div>}
      {error && <div role="alert" className="flex items-start gap-2 rounded-lg border border-[#f0dadc] bg-[#fff8f8] p-3 text-[10px] leading-relaxed text-[#af515e]"><AlertCircle size={13} className="mt-px shrink-0"/>{error}</div>}
      <div ref={endRef}/>
    </div>

    <div className="border-t border-[#eef0f3] bg-white px-4 py-2.5"><p role="status" className="flex items-center gap-1.5 text-[9px] text-[#798591]"><Check size={11}/>Pandas computes supported answers directly from your saved CSV. No generative AI is used.</p></div>
    <div className="border-t border-[#e9ecf0] bg-[#f9fafc] px-4 py-2 text-[9px] leading-relaxed text-[#7d8792]">Privacy: analysis runs on the NEXORA API with the saved dataset you own. The CSV and question are not sent to a hosted LLM. {status?.llm.state === "configured_unverified" ? "Optional Groq credentials are configured but have not been tested or used by this assistant." : statusFailed ? "Optional LLM provider status could not be checked; deterministic analysis remains available." : status ? status.llm.message : "Checking optional hosted LLM configuration…"}</div>
    <form onSubmit={submit} className="flex items-end gap-2 border-t border-[#eef0f3] p-3"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} maxLength={1000} rows={2} placeholder="Ask about row counts, missing values, summaries, categories, or numeric values…" disabled={loading} className="max-h-28 min-h-10 flex-1 resize-y rounded-lg border border-[#e6e9ed] bg-[#fafbfc] px-3 py-2.5 text-[11px] leading-relaxed text-[#46515e] outline-none placeholder:text-[#a1a9b2] focus:border-[#9c90ec] disabled:cursor-not-allowed disabled:opacity-60"/><button type="submit" disabled={loading || !question.trim()} aria-label="Send question" className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-[#6f60df] text-white transition hover:bg-[#5f51ce] disabled:cursor-not-allowed disabled:opacity-45">{loading ? <LoaderCircle size={15} className="animate-spin"/> : <Send size={15}/>}</button></form>
  </section>;
}
