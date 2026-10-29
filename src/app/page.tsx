"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { LoaderCircle } from "lucide-react";
import { api } from "@/lib/auth";

export default function Home() {
  const router = useRouter();
  const [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    api<{ dataset_id: string | null }>("/datasets/active")
      .then(({ dataset_id }) => { if (active) router.replace(dataset_id ? `/datasets/${encodeURIComponent(dataset_id)}` : "/new-analysis"); })
      .catch((cause: unknown) => { if (active) setError(cause instanceof Error ? cause.message : "Could not restore your active dataset."); });
    return () => { active = false; };
  }, [router]);
  return <main className="grid min-h-screen place-items-center text-center text-sm text-[#788390]">{error ? <div role="alert">{error}<p><button className="mt-3 text-[#6f60df]" onClick={() => router.replace("/new-analysis")}>Start a new analysis</button></p></div> : <div><LoaderCircle className="mx-auto animate-spin text-[#6f60df]"/><p className="mt-3">Restoring your active dataset…</p></div>}</main>;
}
