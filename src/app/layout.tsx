import type { Metadata } from "next";
import "./globals.css";
import AuthGate from "@/components/auth-gate";

export const metadata: Metadata = {
  title: "NEXORA AI — Data workspace",
  description: "A local-first workspace for exploring and preparing datasets.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body><AuthGate>{children}</AuthGate></body></html>;
}
