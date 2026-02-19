'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';

const NAV_ITEMS = [
  { href: '/', label: 'Dashboard' },
  { href: '/tickets', label: 'Tickets' },
  { href: '/experiments', label: 'Experiments' },
  { href: '/queue', label: 'Review Queue' },
];

export default function NavHeader() {
  const pathname = usePathname();

  function isActive(href: string) {
    if (href === '/') return pathname === '/';
    return pathname.startsWith(href);
  }

  return (
    <header className="header">
      <h1>CS QA Studio</h1>
      <nav className="nav">
        {NAV_ITEMS.map(({ href, label }) => (
          <Link key={href} href={href} className={isActive(href) ? 'active' : ''}>
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
