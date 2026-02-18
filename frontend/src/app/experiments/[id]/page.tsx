'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { api, Experiment } from '@/lib/api';

export default function ExperimentDetailPage() {
  const params = useParams();
  const expId = params.id as string;

  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const res = await api.getExperiment(expId);
        setExperiment(res);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [expId]);

  if (loading) return <div className="loading">Loading...</div>;
  if (!experiment) return <div className="error">Experiment not found</div>;

  const summary = experiment.summary;

  return (
    <>
      <header className="header">
        <h1>CS QA Studio</h1>
        <nav className="nav">
          <Link href="/">Dashboard</Link>
          <Link href="/tickets">Tickets</Link>
          <Link href="/experiments" className="active">Experiments</Link>
          <Link href="/queue">Review Queue</Link>
        </nav>
      </header>

      <main className="container">
        <div style={{ marginBottom: 16 }}>
          <Link href="/experiments">&larr; Back to Experiments</Link>
        </div>

        <div className="card">
          <h2>{experiment.name}</h2>
          <div style={{ display: 'flex', gap: 16, marginTop: 16, color: 'var(--text-secondary)' }}>
            <span>Split: <strong>{experiment.dataset_split}</strong></span>
            <span>Docs: <strong>{experiment.docs_version}</strong></span>
            <span>Created: <strong>{new Date(experiment.created_at).toLocaleString()}</strong></span>
          </div>
        </div>

        {/* Config Comparison */}
        <div className="grid grid-2" style={{ marginTop: 16 }}>
          <div className="card">
            <h3>Config A</h3>
            <p style={{ marginTop: 8 }}>
              Prompt: <strong>{experiment.config_a.prompt_version}</strong><br />
              Model: <strong>{experiment.config_a.model_version}</strong>
            </p>
          </div>
          <div className="card">
            <h3>Config B</h3>
            <p style={{ marginTop: 8 }}>
              Prompt: <strong>{experiment.config_b.prompt_version}</strong><br />
              Model: <strong>{experiment.config_b.model_version}</strong>
            </p>
          </div>
        </div>

        {/* Summary Stats */}
        {summary && (
          <>
            <div className="card" style={{ marginTop: 16 }}>
              <h2>Results Summary</h2>

              <div className="stats-grid" style={{ marginTop: 16 }}>
                <div className="stat-card">
                  <div className="stat-value">{summary.total_tickets}</div>
                  <div className="stat-label">Total Tickets</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{summary.human_queue_count}</div>
                  <div className="stat-label">Human Queue</div>
                </div>
                <div className="stat-card">
                  <div className="stat-value">{(summary.human_queue_rate * 100).toFixed(1)}%</div>
                  <div className="stat-label">Ambiguous Rate</div>
                </div>
              </div>

              {/* Gate Fail Rates */}
              <div className="grid grid-2" style={{ marginTop: 24 }}>
                <div>
                  <h3 style={{ marginBottom: 12 }}>Gate Fail Rate</h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span>Config A</span>
                        <span>{(summary.gate_fail_rate_a * 100).toFixed(1)}%</span>
                      </div>
                      <div className="score-bar-bg">
                        <div
                          className="score-bar-fill"
                          style={{
                            width: `${summary.gate_fail_rate_a * 100}%`,
                            background: 'var(--error)'
                          }}
                        />
                      </div>
                    </div>
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                        <span>Config B</span>
                        <span>{(summary.gate_fail_rate_b * 100).toFixed(1)}%</span>
                      </div>
                      <div className="score-bar-bg">
                        <div
                          className="score-bar-fill"
                          style={{
                            width: `${summary.gate_fail_rate_b * 100}%`,
                            background: 'var(--error)'
                          }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* Average Scores */}
                <div>
                  <h3 style={{ marginBottom: 12 }}>Average Scores</h3>
                  <table className="table">
                    <thead>
                      <tr>
                        <th>Score</th>
                        <th>A</th>
                        <th>B</th>
                        <th>Diff</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.keys(summary.avg_scores_a).map(key => {
                        const a = summary.avg_scores_a[key] || 0;
                        const b = summary.avg_scores_b[key] || 0;
                        const diff = a - b;
                        return (
                          <tr key={key}>
                            <td>{key}</td>
                            <td>{a.toFixed(2)}</td>
                            <td>{b.toFixed(2)}</td>
                            <td style={{ color: diff > 0 ? 'var(--success)' : diff < 0 ? 'var(--error)' : 'inherit' }}>
                              {diff > 0 ? '+' : ''}{diff.toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>

              {/* Tag Delta */}
              {Object.keys(summary.top_tag_delta).length > 0 && (
                <div style={{ marginTop: 24 }}>
                  <h3 style={{ marginBottom: 12 }}>Tag Delta (A - B)</h3>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {Object.entries(summary.top_tag_delta)
                      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                      .map(([tag, delta]) => (
                        <span
                          key={tag}
                          className="tag"
                          style={{
                            background: delta > 0 ? 'rgba(239, 68, 68, 0.2)' : 'rgba(34, 197, 94, 0.2)',
                            color: delta > 0 ? 'var(--error)' : 'var(--success)'
                          }}
                        >
                          {tag}: {delta > 0 ? '+' : ''}{delta}
                        </span>
                      ))}
                  </div>
                </div>
              )}

              {/* Actionability Distribution */}
              <div className="grid grid-2" style={{ marginTop: 24 }}>
                <div>
                  <h3 style={{ marginBottom: 12 }}>Actionability Distribution (A)</h3>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {[1, 2, 3, 4, 5].map(score => (
                      <div key={score} style={{ flex: 1, textAlign: 'center' }}>
                        <div className={`badge badge-${score <= 2 ? 'error' : score >= 4 ? 'success' : 'warning'}`}>
                          {score}
                        </div>
                        <div style={{ marginTop: 4 }}>
                          {summary.actionability_distribution_a[score] || 0}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 style={{ marginBottom: 12 }}>Actionability Distribution (B)</h3>
                  <div style={{ display: 'flex', gap: 8 }}>
                    {[1, 2, 3, 4, 5].map(score => (
                      <div key={score} style={{ flex: 1, textAlign: 'center' }}>
                        <div className={`badge badge-${score <= 2 ? 'error' : score >= 4 ? 'success' : 'warning'}`}>
                          {score}
                        </div>
                        <div style={{ marginTop: 4 }}>
                          {summary.actionability_distribution_b[score] || 0}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </>
        )}
      </main>
    </>
  );
}
