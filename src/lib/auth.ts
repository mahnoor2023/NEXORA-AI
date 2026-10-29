"use client";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");
export type User = { id: string; email: string; display_name: string; created_at: string };
export type DatasetRecord = { id: string; filename: string; size_bytes: number; row_count: number; column_count: number; created_at: string; updated_at: string; last_opened_at: string | null; available: boolean; last_kind: string | null };
export type HistoryRecord = { id: string; dataset_id: string; filename: string; kind: string; summary: Record<string, unknown>; created_at: string };
export type Preferences = { default_view: "overview" | "datasets" | "exploration"; compact_tables: boolean };

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_URL}${path}`, { ...init, credentials: "include", headers: { ...(init.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}), ...init.headers } });
  const result = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = result?.detail;
    const message = Array.isArray(detail) ? detail.map((item: { msg?: string }) => item.msg).filter(Boolean).join(" ") : null;
    throw new Error(typeof detail === "string" ? detail : detail?.message ?? message ?? `Request failed (${response.status})`);
  }
  return result as T;
}
