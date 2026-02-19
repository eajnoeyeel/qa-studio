'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, Ticket } from '@/lib/api';

export default function TicketsPage() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [split, setSplit] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadTickets() {
      setLoading(true);
      try {
        const res = await api.listTickets(page, 20, split || undefined);
        setTickets(res.tickets);
        setTotal(res.total);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadTickets();
  }, [page, split]);

  return (
      <main className="container">
        <div className="card">
          <div className="card-header">
            <h2>Tickets ({total})</h2>
            <div style={{ display: 'flex', gap: 12 }}>
              <select
                className="select"
                value={split}
                onChange={(e) => { setSplit(e.target.value); setPage(1); }}
              >
                <option value="">All Splits</option>
                <option value="dev">dev</option>
                <option value="test">test</option>
                <option value="ab_eval">ab_eval</option>
              </select>
            </div>
          </div>

          {loading ? (
            <div className="loading">Loading...</div>
          ) : (
            <>
              <table className="table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Split</th>
                    <th>Messages</th>
                    <th>Response Preview</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {tickets.map((ticket) => (
                    <tr key={ticket.id}>
                      <td>
                        <Link href={`/tickets/${ticket.id}`}>
                          {ticket.external_id || ticket.id.slice(0, 8)}
                        </Link>
                      </td>
                      <td><span className="badge badge-info">{ticket.split}</span></td>
                      <td>{ticket.conversation.length}</td>
                      <td style={{ maxWidth: 300, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {ticket.candidate_response.slice(0, 100)}...
                      </td>
                      <td>{new Date(ticket.created_at).toLocaleDateString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>

              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 16 }}>
                <button
                  className="btn btn-secondary"
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                  disabled={page === 1}
                >
                  Previous
                </button>
                <span style={{ color: 'var(--text-secondary)' }}>Page {page}</span>
                <button
                  className="btn btn-secondary"
                  onClick={() => setPage(p => p + 1)}
                  disabled={tickets.length < 20}
                >
                  Next
                </button>
              </div>
            </>
          )}
        </div>
      </main>
  );
}
