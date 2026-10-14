import type { Metadata } from "next";
import "./globals.css";
import React from "react";
import { NuqsAdapter } from "nuqs/adapters/next/app";

// System font stack instead of next/font/google: the image must build without internet access.
const fontStack =
  'Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';

export const metadata: Metadata = {
  title: "Apertus Interview Coach",
  description: "Practice apprenticeship interviews with an Apertus-based coach (HackApertus 2026)",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="de">
      <body style={{ fontFamily: fontStack }}>
        <NuqsAdapter>{children}</NuqsAdapter>
      </body>
    </html>
  );
}
