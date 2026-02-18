'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, Ticket, Experiment } from '@/lib/api';

export default function Home() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [stats, setStats] = useState<{
    total_evaluations: number;
    gate_fail_rate: number;
    avg_scores: Record<string, number>;
  } | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        const [ticketRes, expRes, reportRes] = await Promise.all([
          api.listTickets(1, 10),
          api.listExperiments(),
          api.getReportSummary('dev').catch(() => null),
        ]);
        setTickets(ticketRes.tickets);
        setExperiments(expRes.experiments);
        if (reportRes) setStats(reportRes);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">{error}</div>;

  return (
    <>
      <header className="header">
        <h1>CS QA Studio</h1>
        <nav className="nav">
          <Link href="/" className="active">Dashboard</Link>
          <Link href="/tickets">Tickets</Link>
          <Link href="/experiments">Experiments</Link>
          <Link href="/queue">Review Queue</Link>
        </nav>
      </header>

      <main className="container">
        {/* Stats Overview */}
        {stats && (
          <div className="card">
            <h2>Overview (dev split)</h2>
            <div className="stats-grid" style={{ marginTop: 16 }}>
              <div className="stat-card">
                <div className="stat-value">{stats.total_evaluations}</div>
                <div className="stat-label">Total Evaluations</div>
              </div>
              <div className="stat-card">
                <div className="stat-value" style={{ color: stats.gate_fail_rate > 0.1 ? 'var(--error)' : 'var(--success)' }}>
                  {(stats.gate_fail_rate * 100).toFixed(1)}%
                </div>
                <div className="stat-label">Gate Fail Rate</div>
              </div>
              {Object.entries(stats.avg_scores).map(([key, value]) => (
                <div className="stat-card" key={key}>
                  <div className="stat-value">{value.toFixed(2)}</div>
                  <div className="stat-label">{key.replace('_', ' ')}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        <div className="grid grid-2">
          {/* Recent Tickets */}
          <div className="card">
            <div className="card-header">
              <h2>Recent Tickets</h2>
              <Link href="/tickets" className="btn btn-secondary">View All</Link>
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Split</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {tickets.slice(0, 5).map((ticket) => (
                  <tr key={ticket.id}>
                    <td>
                      <Link href={`/tickets/${ticket.id}`}>
                        {ticket.external_id || ticket.id.slice(0, 8)}
                      </Link>
                    </td>
                    <td><span className="badge badge-info">{ticket.split}</span></td>
                    <td>{new Date(ticket.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Recent Experiments */}
          <div className="card">
            <div className="card-header">
              <h2>Recent Experiments</h2>
              <Link href="/experiments" className="btn btn-secondary">View All</Link>
            </div>
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {experiments.slice(0, 5).map((exp) => (
                  <tr key={exp.id}>
                    <td>
                      <Link href={`/experiments/${exp.id}`}>{exp.name}</Link>
                    </td>
                    <td>
                      <span className={`badge ${exp.completed_at ? 'badge-success' : 'badge-warning'}`}>
                        {exp.completed_at ? 'Completed' : 'Running'}
                      </span>
                    </td>
                    <td>{new Date(exp.created_at).toLocaleDateString()}</td>
                  </tr>
                ))}
                {experiments.length === 0 && (
                  <tr>
                    <td colSpan={3} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                      No experiments yet
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </main>
    </>
  );
}
