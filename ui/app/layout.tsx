import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "FMODetect — research demo",
  description:
    "PyTorch re-implementation of FMODetect with CBAM attention, joint TDF + matting head, and uncertainty-weighted boundary loss.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen selection:bg-[--accent-soft]/40 selection:text-[--text]">
        {children}
      </body>
    </html>
  );
}
