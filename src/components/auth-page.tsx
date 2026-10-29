"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";
import { api, type User } from "@/lib/auth";
import { LoaderCircle, Sparkles } from "lucide-react";

export default function AuthPage({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const submit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault(); setError(""); setLoading(true);
    const form = new FormData(event.currentTarget);
    const payload = Object.fromEntries(form.entries());
    try { await api<{ user: User }>(`/auth/${mode}`, { method: "POST", body: JSON.stringify(payload) }); router.replace("/"); router.refresh(); }
    catch (cause) { setError(cause instanceof Error ? cause.message : "Could not authenticate."); }
    finally { setLoading(false); }
  };
  return <main className="grid min-h-screen place-items-center bg-[#f7f8fb] px-4 py-10"><section className="w-full max-w-[420px] rounded-2xl border border-[#e8ebef] bg-white p-7 shadow-sm"><div className="mb-6 flex items-center gap-2.5"><span className="grid h-9 w-9 place-items-center rounded-xl bg-[#6f61df] text-white"><Sparkles size={17}/></span><b className="text-[17px]">NEXORA <span className="text-[#7566e9]">AI</span></b></div><h1 className="text-2xl font-bold text-[#252e39]">{mode === "signup" ? "Create your account" : "Welcome back"}</h1><p className="mt-2 text-sm text-[#7f8995]">{mode === "signup" ? "Your datasets and analysis history are private to your account." : "Sign in to continue to your data workspace."}</p><form className="mt-6 space-y-4" onSubmit={submit}>
    {mode === "signup" && <label className="block text-xs font-semibold text-[#596574]">Name<input name="display_name" required minLength={1} maxLength={80} autoComplete="name" className="mt-1.5 h-10 w-full rounded-lg border border-[#e2e6ea] px-3 text-sm font-normal outline-none focus:border-[#8c7fe8]"/></label>}
    <label className="block text-xs font-semibold text-[#596574]">Email<input name="email" required type="email" maxLength={254} autoComplete="email" className="mt-1.5 h-10 w-full rounded-lg border border-[#e2e6ea] px-3 text-sm font-normal outline-none focus:border-[#8c7fe8]"/></label>
    <label className="block text-xs font-semibold text-[#596574]">Password<input name="password" required type="password" minLength={mode === "signup" ? 12 : 1} maxLength={128} autoComplete={mode === "signup" ? "new-password" : "current-password"} className="mt-1.5 h-10 w-full rounded-lg border border-[#e2e6ea] px-3 text-sm font-normal outline-none focus:border-[#8c7fe8]"/>{mode === "signup" && <span className="mt-1 block font-normal text-[#909aa5]">Use at least 12 characters.</span>}</label>
    {error && <p role="alert" className="rounded-lg bg-[#fff3f2] px-3 py-2 text-sm text-[#b34c56]">{error}</p>}
    <button disabled={loading} className="flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-[#6f60df] text-sm font-semibold text-white disabled:opacity-60">{loading && <LoaderCircle size={15} className="animate-spin"/>}{mode === "signup" ? "Create account" : "Sign in"}</button>
  </form><p className="mt-5 text-center text-sm text-[#7f8995]">{mode === "signup" ? "Already have an account? " : "New to NEXORA AI? "}<Link className="font-semibold text-[#695bd5]" href={mode === "signup" ? "/login" : "/signup"}>{mode === "signup" ? "Sign in" : "Create an account"}</Link></p></section></main>;
}
