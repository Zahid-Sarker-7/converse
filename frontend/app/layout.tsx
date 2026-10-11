import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Converse — AI role-play & coaching",
  description:
    "Practice interviews and conversations with an in-character AI, then get scored on the moments that mattered.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <div className="aurora" aria-hidden />
        {children}
      </body>
    </html>
  );
}
