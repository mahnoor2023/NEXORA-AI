import type { ReactNode } from "react";
import { AppNav } from "@/components/app-nav";
export default function AppFrame({ children }: { children: ReactNode }) { return <><AppNav/><div className="min-h-screen pb-16 md:pb-0 md:pl-[248px]">{children}</div></>; }
