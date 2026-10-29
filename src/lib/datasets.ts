"use client";

import { api } from "@/lib/auth";
import type { DatasetAnalysis } from "@/types/dataset";

const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1").replace(/\/$/, "");

export function reopenDataset(datasetId: string): Promise<DatasetAnalysis> {
  return api<DatasetAnalysis>(`/datasets/${encodeURIComponent(datasetId)}/analysis`);
}

export async function downloadDatasetSource(datasetId: string, filename: string): Promise<File> {
  const response = await fetch(`${API_URL}/datasets/${encodeURIComponent(datasetId)}/source`, { credentials: "include" });
  if (!response.ok) {
    const result = await response.json().catch(() => null);
    const detail = result?.detail;
    throw new Error(typeof detail === "string" ? detail : detail?.message ?? `Could not reopen dataset (${response.status})`);
  }
  const csv = await response.blob();
  return new File([csv], filename, { type: "text/csv", lastModified: Date.now() });
}
