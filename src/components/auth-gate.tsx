"use client";

import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";
import { api, type User } from "@/lib/auth";
import { LoaderCircle } from "lucide-react";

export default function AuthGate({ children }: { children: ReactNode }) {
  const path = usePathname();
  const router = useRouter();
  const isPublic = path === "/login" || path === "/signup";
  const [ready, setReady] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  useEffect(() => {
    let alive = true;
    api<{ user: User }>("/auth/me").then(({ user: value }) => { if (alive) setUser(value); }).catch(() => { if (alive) setUser(null); }).finally(() => { if (alive) setReady(true); });
    return () => { alive = false; };
  }, [path]);
  useEffect(() => {
    if (!ready) return;
    if (!isPublic && !user) router.replace("/login");
    if (isPublic && user) router.replace("/");
  }, [ready, isPublic, user, router]);
  if (!ready || (!isPublic && !user) || (isPublic && user)) return <div className="grid min-h-screen place-items-center text-[#6f60df]"><LoaderCircle className="animate-spin"/></div>;
  return <>{children}</>;
}
