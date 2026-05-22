import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "FMODetect-v2",
  description: "Fast Moving Object detection with CBAM + multi-task + uncertainty.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
