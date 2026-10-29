import type { AgentChatResult, AnalystStatus, DatasetAnalysis, MLTask, MLTrainingResult } from "@/types/dataset";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

function errorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail && typeof detail.message === "string") return detail.message;
  if (Array.isArray(detail)) return detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join(" ") || fallback;
  return fallback;
}

export async function analyzeCsv(file: File, cleaning?: { drop_duplicates: boolean; fill_numeric: boolean; fill_categorical: boolean }): Promise<DatasetAnalysis> {
  const body = new FormData();
  body.append("file", file);
  if (cleaning) body.append("cleaning", JSON.stringify(cleaning));
  const response = await fetch(`${API_URL}/datasets/analyze`, { method: "POST", body, credentials: "include" });
  if (!response.ok) {
    const result = await response.json().catch(() => null);
    throw new Error(errorMessage(result?.detail, `Analysis failed (${response.status})`));
  }
  return response.json() as Promise<DatasetAnalysis>;
}

export async function trainCsv(file: File, target: string, task: MLTask): Promise<MLTrainingResult> {
  const body = new FormData();
  body.append("file", file);
  body.append("target", target);
  body.append("task", task);
  const response = await fetch(`${API_URL}/datasets/train`, { method: "POST", body, credentials: "include" });
  if (!response.ok) {
    const result = await response.json().catch(() => null);
    throw new Error(errorMessage(result?.detail, `Training failed (${response.status})`));
  }
  return response.json() as Promise<MLTrainingResult>;
}

export async function getAnalystStatus(): Promise<AnalystStatus> {
  const response = await fetch(`${API_URL}/agent/status`, { cache: "no-store", credentials: "include" });
  if (!response.ok) throw new Error(`Agent status unavailable (${response.status})`);
  return response.json() as Promise<AnalystStatus>;
}

export class AgentRequestError extends Error {
  constructor(message: string, public readonly code: string) {
    super(message);
    this.name = "AgentRequestError";
  }
}

export async function askAnalyst(file: File, question: string): Promise<AgentChatResult> {
  const body = new FormData();
  body.append("file", file);
  body.append("question", question);
  const response = await fetch(`${API_URL}/agent/chat`, { method: "POST", body, credentials: "include" });
  if (!response.ok) {
    const result = await response.json().catch(() => null);
    const detail = result?.detail;
    if (detail && typeof detail === "object") {
      throw new AgentRequestError(detail.message ?? `Agent request failed (${response.status})`, detail.code ?? "AGENT_ANALYSIS_ERROR");
    }
    throw new AgentRequestError(typeof detail === "string" ? detail : `Agent request failed (${response.status})`, "AGENT_ANALYSIS_ERROR");
  }
  return response.json() as Promise<AgentChatResult>;
}
