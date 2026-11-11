import { api } from "@/lib/auth";
import type { BIProfile, BIResult, BIMapping } from "@/types/business-intelligence";

export function getBIProfile(datasetId: string): Promise<BIProfile> {
  return api<BIProfile>(`/datasets/${encodeURIComponent(datasetId)}/business-intelligence/profile`);
}

export function analyzeBI(datasetId: string, mapping: BIMapping, startDate: string, endDate: string): Promise<BIResult> {
  return api<BIResult>(`/datasets/${encodeURIComponent(datasetId)}/business-intelligence/analyze`, {
    method: "POST",
    body: JSON.stringify({ mapping, start_date: startDate || null, end_date: endDate || null }),
  });
}
