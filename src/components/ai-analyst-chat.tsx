"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";
import { AlertCircle, Bot, Check, CircleHelp, LoaderCircle, Send, ShieldCheck, Sparkles, Wifi, WifiOff } from "lucide-react";
import { AgentRequestError, askAnalyst, getAnalystStatus } from "@/lib/api";
import type { AgentChatResult, AnalystStatus } from "@/types/dataset";

type ChatMessage = { role: "user"; text: string } | { role: "assistant"; result: AgentChatResult };

function statusDescription(status: AnalystStatus | null, failed: boolean) {
  if (failed) return "API status could not be checked. Confirm that FastAPI is running.";
  if (!status) return "Checking AI provider configuration…";
  if (status.state === "ready") return status.message;
  if (status.state === "missing_credentials") return "Hosted AI is not configured. Set GROQ_API_KEY on the backend server.";
  if (status.state === "rate_limited") return "The AI provider is rate-limited. Wait briefly and retry.";
  if (status.state === "unavailable" || status.state === "request_failed") return status.message;
  if (status.state === "ollama_unavailable") return "Neither GroqCloud nor a configured local Ollama model is ready.";
  return status.message;
}

function providerName(provider: "groq" | "ollama_local" | undefined) {
  return provider === "groq" ? "GroqCloud" : "Ollama";
}

export default function AIAnalystChat({ file }: { file: File }) {
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
      const result = await askAnalyst(file, prompt);
      setMessages((current) => [...current, { role: "assistant", result }]);
    } catch (cause) {
      const agentError = cause instanceof AgentRequestError ? cause : null;
      const detail = cause instanceof Error ? cause.message : "The analysis could not be completed.";
      const messagesByCode: Record<string, string> = {
        GROQ_RATE_LIMITED: "Groq rate limit reached. Wait briefly and retry.",
        GROQ_MISSING_CREDENTIALS: "Hosted AI is not configured. Set GROQ_API_KEY on the backend server.",
        GROQ_UNAVAILABLE: "GroqCloud is unavailable. Check provider status and retry.",
        OLLAMA_UNAVAILABLE: "Ollama became unavailable. Check the local service and retry.",
        AGENT_TIMEOUT: "The AI provider took too long. Try a shorter question or smaller dataset.",
        TOOL_CALL_LIMIT: "The agent used its analysis-call budget. Try a more focused question.",
      };
      setError(agentError ? messagesByCode[agentError.code] ?? detail : detail);
      if (agentError?.code.startsWith("OLLAMA_") || agentError?.code.startsWith("GROQ_")) {
        getAnalystStatus().then(setStatus).catch(() => setStatusFailed(true));
      }
    } finally {
      setLoading(false);
    }
  };

  const ready = status?.state === "ready";
  const description = statusDescription(status, statusFailed);

  return <section className="overflow-hidden rounded-xl border border-[#e9ecf0] bg-white">
    <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#eef0f3] px-5 py-4">
      <div className="flex items-center gap-2.5"><span className="grid h-9 w-9 place-items-center rounded-[10px] bg-[#f0edff] text-[#6f60df]"><Bot size={18}/></span><div><h3 className="text-[13px] font-bold text-[#303946]">AI Data Analyst</h3><p className="mt-0.5 text-[10px] text-[#99a2ac]">Evidence-grounded chat · {file.name}</p></div></div>
      <span className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] font-semibold ${ready ? "bg-[#eaf8f4] text-[#398f7b]" : statusFailed ? "bg-[#fff1f0] text-[#b45c5c]" : "bg-[#fff6e9] text-[#aa7d3d]"}`}>
        {ready ? <Wifi size={11}/> : status || statusFailed ? <WifiOff size={11}/> : <LoaderCircle size={11} className="animate-spin"/>}
        {ready ? `${providerName(status?.provider)} · ${status?.model}` : status?.state === "missing_credentials" ? "Credentials required" : status?.state === "rate_limited" ? "Rate limited" : status?.state === "unavailable" ? "Provider unavailable" : status?.state === "request_failed" || status?.state === "configuration_error" ? "Provider error" : status?.state === "model_unavailable" ? "Model unavailable" : statusFailed || status?.state === "ollama_unavailable" ? "Provider unavailable" : "Checking"}
      </span>
    </div>

    <div aria-live="polite" className="scrollbar-hide flex min-h-[280px] max-h-[520px] flex-col gap-3 overflow-y-auto bg-[#fafbfc] p-4 sm:p-5">
      {!messages.length && <div className="m-auto max-w-[420px] py-8 text-center"><span className="mx-auto grid h-11 w-11 place-items-center rounded-xl bg-white text-[#7869e6] shadow-sm"><Sparkles size={19}/></span><h4 className="mt-3 text-[12px] font-bold text-[#485361]">Ask about this dataset</h4><p className="mt-1.5 text-[10px] leading-relaxed text-[#929ca7]">For example: “Which columns have the most missing data?” or “Summarize the trained model results.” Only bounded computed evidence is sent to the configured AI provider; raw rows are not sent.</p></div>}
      {messages.map((message, index) => message.role === "user"
        ? <div key={index} className="ml-auto max-w-[85%] rounded-xl rounded-br-sm bg-[#6f60df] px-3.5 py-2.5 text-[11px] leading-relaxed text-white">{message.text}</div>
        : <article key={index} className="max-w-[92%] rounded-xl rounded-bl-sm border border-[#e9ecf0] bg-white p-3.5 shadow-[0_2px_9px_#26303b06]"><div className="mb-2 flex items-center gap-1.5 text-[9px] font-semibold text-[#7062d8]"><Bot size={12}/>{providerName(message.result.provider)} · {message.result.model}</div><p className="whitespace-pre-wrap text-[11px] leading-relaxed text-[#4e5a67]">{message.result.answer}</p><div className="mt-3 flex flex-wrap gap-1.5">{message.result.tools_used.map((tool) => <span key={tool} className="rounded bg-[#f4f5f8] px-2 py-1 text-[8px] text-[#77828e]">{tool}</span>)}</div><details className="mt-2.5"><summary className="cursor-pointer text-[9px] font-semibold text-[#7c8793]">Evidence citations ({message.result.citations.length})</summary><ul className="mt-2 space-y-1.5">{message.result.citations.map((citation, citeIndex) => <li key={`${citation.tool}-${citeIndex}`} className="rounded-md bg-[#f8f9fb] p-2 text-[9px] leading-relaxed text-[#6f7b88]"><span className="font-semibold text-[#7163d6]">{citation.tool}: </span>{citation.quote}</li>)}</ul></details><p className="mt-2 flex items-start gap-1.5 text-[8px] leading-relaxed text-[#a0a8b1]"><ShieldCheck size={10} className="mt-px shrink-0 text-[#54a994]"/>{message.result.privacy}</p></article>)}
      {loading && <div className="flex w-fit items-center gap-2 rounded-lg border border-[#e9ecf0] bg-white px-3 py-2 text-[10px] text-[#7e8995]"><LoaderCircle size={13} className="animate-spin text-[#7667e4]"/>Checking evidence and consulting the configured AI provider…</div>}
      {error && <div role="alert" className="flex items-start gap-2 rounded-lg border border-[#f0dadc] bg-[#fff8f8] p-3 text-[10px] leading-relaxed text-[#af515e]"><AlertCircle size={13} className="mt-px shrink-0"/>{error}</div>}
      <div ref={endRef}/>
    </div>

    <div className={`border-t px-4 py-2.5 ${ready ? "border-[#eef0f3] bg-white" : "border-[#f0e3c9] bg-[#fffaf2]"}`}><p role="status" className={`flex items-center gap-1.5 text-[9px] ${ready ? "text-[#798591]" : "text-[#987644]"}`}>{ready ? <Check size={11}/> : <CircleHelp size={11}/>} {description}</p></div>
    <div className="border-t border-[#e9ecf0] bg-[#f9fafc] px-4 py-2 text-[9px] leading-relaxed text-[#7d8792]">{status?.provider === "ollama_local" ? "Privacy notice: the selected schema, aggregate statistics, tool results, and your question are processed by your configured local Ollama service. Raw CSV rows are not sent." : "Privacy notice: selected schema, aggregate statistics, tool results, and your question are processed by the configured third-party cloud AI provider. Raw CSV rows are not sent. Configure Ollama to keep this processing local."}</div>
    <form onSubmit={submit} className="flex items-end gap-2 border-t border-[#eef0f3] p-3"><textarea value={question} onChange={(event) => setQuestion(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} maxLength={1000} rows={2} placeholder={ready ? "Ask a question about your data…" : "Configure an AI provider to chat"} disabled={!ready || loading} className="max-h-28 min-h-10 flex-1 resize-y rounded-lg border border-[#e6e9ed] bg-[#fafbfc] px-3 py-2.5 text-[11px] leading-relaxed text-[#46515e] outline-none placeholder:text-[#a1a9b2] focus:border-[#9c90ec] disabled:cursor-not-allowed disabled:opacity-60"/><button type="submit" disabled={!ready || loading || !question.trim()} aria-label="Send question" className="grid h-10 w-10 shrink-0 place-items-center rounded-lg bg-[#6f60df] text-white transition hover:bg-[#5f51ce] disabled:cursor-not-allowed disabled:opacity-45">{loading ? <LoaderCircle size={15} className="animate-spin"/> : <Send size={15}/>}</button></form>
  </section>;
}
