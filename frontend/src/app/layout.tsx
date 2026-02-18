import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'CS QA Studio',
  description: 'SaaS Customer Service Quality Assessment Studio',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
