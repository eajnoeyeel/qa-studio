'use client';

import { useEffect, useState } from 'react';
import { api, Proposal, CycleResult, ExperimentSummary } from '@/lib/api';

const STATUS_COLORS: Record<string, string> = {
  pending: 'badge-warning',
  testing: 'badge-info',
  approved: 'badge-success',
  rejected: 'badge-error',
  deployed: 'badge-success',
};

export default function ImprovePage() {
  const [proposals, setProposals] = useState<Proposal[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [cycleResult, setCycleResult] = useState<CycleResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [experimentCache, setExperimentCache] = useState<Record<string, ExperimentSummary>>({});

  const [formData, setFormData] = useState({
    dataset_split: 'dev',
    limit: 50,
  });

  useEffect(() => {
    loadProposals();
  }, []);

  async function loadProposals() {
    try {
      const res = await api.listProposals();
      setProposals(res);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function loadExperiment(experimentId: string) {
    if (experimentCache[experimentId]) return;
    try {
      const exp = await api.getExperiment(experimentId);
      if (exp.summary) {
        setExperimentCache((prev) => ({ ...prev, [experimentId]: exp.summary! }));
      }
    } catch (err) {
      console.error(err);
    }
  }

  async function runCycle() {
    setRunning(true);
    setError(null);
    setCycleResult(null);
    try {
      const result = await api.runImprovementCycle({
        dataset_split: formData.dataset_split,
        limit: formData.limit,
      });
      setCycleResult(result);
      loadProposals();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Cycle failed');
    } finally {
      setRunning(false);
    }
  }

  async function handleDeploy(id: string) {
    try {
      await api.deployProposal(id);
      loadProposals();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Deploy failed');
    }
  }

  async function handleApprove(id: string) {
    try {
      await api.approveProposal(id);
      loadProposals();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Approve failed');
    }
  }

  function toggleExpand(p: Proposal) {
    if (expandedId === p.id) {
      setExpandedId(null);
    } else {
      setExpandedId(p.id);
      if (p.test_experiment_id) {
        loadExperiment(p.test_experiment_id);
      }
    }
  }

  function renderScoreComparison(summary: ExperimentSummary) {
    const keys = [...new Set([
      ...Object.keys(summary.avg_scores_a),
      ...Object.keys(summary.avg_scores_b),
    ])];
    if (keys.length === 0) return <p style={{ color: 'var(--text-secondary)' }}>No scores available</p>;

    return (
      <table className="table" style={{ fontSize: '0.85rem' }}>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Baseline</th>
            <th>Candidate</th>
            <th>Delta</th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => {
            const a = summary.avg_scores_a[k] ?? 0;
            const b = summary.avg_scores_b[k] ?? 0;
            const delta = b - a;
            const color = delta > 0 ? '#22c55e' : delta < 0 ? '#ef4444' : 'inherit';
            return (
              <tr key={k}>
                <td>{k.replace(/_/g, ' ')}</td>
                <td>{a.toFixed(2)}</td>
                <td>{b.toFixed(2)}</td>
                <td style={{ color, fontWeight: delta !== 0 ? 600 : 400 }}>
                  {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    );
  }

  return (
    <main className="container">
      {/* Run Cycle */}
      <div className="card">
        <h2>Self-Improvement Cycle</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: 8 }}>
          Analyzes failure patterns, generates an improved system prompt, runs A/B evaluation, and creates a proposal.
        </p>

        <div className="grid grid-4" style={{ marginTop: 16, gap: 12 }}>
          <select
            className="select"
            value={formData.dataset_split}
            onChange={(e) => setFormData({ ...formData, dataset_split: e.target.value })}
          >
            <option value="dev">dev</option>
            <option value="test">test</option>
            <option value="ab_eval">ab_eval</option>
          </select>
          <input
            className="input"
            type="number"
            placeholder="Limit"
            value={formData.limit}
            onChange={(e) => setFormData({ ...formData, limit: Number(e.target.value) })}
            min={1}
            max={5000}
          />
          <button
            className="btn btn-primary"
            onClick={runCycle}
            disabled={running}
          >
            {running ? 'Running...' : 'Run Cycle'}
          </button>
          <div></div>
        </div>

        {error && (
          <div style={{ marginTop: 12, padding: '8px 12px', background: 'var(--bg-error, #ff4444)', color: '#fff', borderRadius: 6, fontSize: '0.875rem' }}>
            {error}
          </div>
        )}

        {cycleResult && (
          <div style={{ marginTop: 16, padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
            <h3>Cycle Results</h3>
            <div style={{ marginTop: 12, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
              <div><strong>Patterns found:</strong> {cycleResult.patterns_found}</div>
              <div><strong>Proposal ID:</strong> {cycleResult.proposal_id.slice(0, 8)}...</div>
              <div style={{ gridColumn: '1 / -1' }}>
                <strong>Rationale:</strong> {cycleResult.suggestion_rationale}
              </div>
              {cycleResult.langfuse_experiment_url && (
                <div style={{ gridColumn: '1 / -1' }}>
                  <a
                    href={cycleResult.langfuse_experiment_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    style={{ color: 'var(--primary)' }}
                  >
                    View in Langfuse
                  </a>
                </div>
              )}
            </div>

            {/* Inline score comparison from cycle result */}
            {Object.keys(cycleResult.avg_scores_baseline).length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h4>Score Comparison</h4>
                <table className="table" style={{ fontSize: '0.85rem', marginTop: 8 }}>
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th>Baseline</th>
                      <th>Candidate</th>
                      <th>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(cycleResult.avg_scores_baseline).map((k) => {
                      const a = cycleResult.avg_scores_baseline[k] ?? 0;
                      const b = cycleResult.avg_scores_candidate[k] ?? 0;
                      const delta = b - a;
                      const color = delta > 0 ? '#22c55e' : delta < 0 ? '#ef4444' : 'inherit';
                      return (
                        <tr key={k}>
                          <td>{k.replace(/_/g, ' ')}</td>
                          <td>{a.toFixed(2)}</td>
                          <td>{b.toFixed(2)}</td>
                          <td style={{ color, fontWeight: delta !== 0 ? 600 : 400 }}>
                            {delta > 0 ? '+' : ''}{delta.toFixed(2)}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>

      {/* Proposals List */}
      <div className="card" style={{ marginTop: 16 }}>
        <h2>Proposals</h2>
        {loading ? (
          <div className="loading">Loading...</div>
        ) : (
          <div style={{ marginTop: 16 }}>
            {proposals.map((p) => {
              const isExpanded = expandedId === p.id;
              const summary = p.test_experiment_id ? experimentCache[p.test_experiment_id] : null;

              return (
                <div
                  key={p.id}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: 8,
                    marginBottom: 12,
                    overflow: 'hidden',
                  }}
                >
                  {/* Header row */}
                  <div
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 12,
                      padding: '12px 16px',
                      cursor: 'pointer',
                      background: isExpanded ? 'var(--bg-tertiary)' : 'transparent',
                    }}
                    onClick={() => toggleExpand(p)}
                  >
                    <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'monospace' }}>
                      {p.id.slice(0, 8)}
                    </span>
                    <span className={`badge ${STATUS_COLORS[p.status] || ''}`}>
                      {p.status}
                    </span>
                    <span style={{ flex: 1 }}>{p.prompt_name}</span>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                      {p.created_by} &middot; {new Date(p.created_at).toLocaleDateString()}
                    </span>
                    <span style={{ fontSize: '0.8rem' }}>{isExpanded ? '▲' : '▼'}</span>
                  </div>

                  {/* Expanded details */}
                  {isExpanded && (
                    <div style={{ padding: '0 16px 16px' }}>
                      {/* Proposed prompt */}
                      <div style={{ marginTop: 8 }}>
                        <strong>Proposed System Prompt:</strong>
                        <pre
                          style={{
                            marginTop: 8,
                            padding: 12,
                            background: 'var(--bg-secondary, #1a1a2e)',
                            borderRadius: 6,
                            fontSize: '0.85rem',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            lineHeight: 1.5,
                            maxHeight: 300,
                            overflowY: 'auto',
                          }}
                        >
                          {p.proposed_prompt}
                        </pre>
                      </div>

                      {/* Experiment scores */}
                      {summary && (
                        <div style={{ marginTop: 16 }}>
                          <strong>A/B Experiment Scores</strong> ({summary.total_items} items)
                          <div style={{ marginTop: 8 }}>
                            {renderScoreComparison(summary)}
                          </div>
                        </div>
                      )}

                      {p.test_experiment_id && !summary && (
                        <p style={{ marginTop: 12, color: 'var(--text-secondary)', fontSize: '0.85rem' }}>
                          Loading experiment scores...
                        </p>
                      )}

                      {/* Actions */}
                      <div style={{ marginTop: 16, display: 'flex', gap: 8 }}>
                        {p.status === 'testing' && (
                          <button className="btn btn-secondary" onClick={() => handleApprove(p.id)}>
                            Approve
                          </button>
                        )}
                        {p.status === 'approved' && (
                          <button className="btn btn-primary" onClick={() => handleDeploy(p.id)}>
                            Deploy to Langfuse
                          </button>
                        )}
                        {p.test_experiment_id && (
                          <a
                            href={`/experiments/${p.test_experiment_id}`}
                            className="btn btn-secondary"
                            style={{ textDecoration: 'none' }}
                          >
                            View Experiment
                          </a>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}

            {proposals.length === 0 && (
              <p style={{ textAlign: 'center', color: 'var(--text-secondary)', padding: 32 }}>
                No proposals yet. Run an improvement cycle above!
              </p>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
