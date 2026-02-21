import type { Metadata } from 'next';
import NavHeader from '@/components/NavHeader';
import './globals.css';

export const metadata: Metadata = {
  title: 'QA Studio',
  description: 'Automated Q/A Response Quality Assessment Studio',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <NavHeader />
        {children}
      </body>
    </html>
  );
}
