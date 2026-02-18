'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, Experiment, ExperimentSummary } from '@/lib/api';

export default function ExperimentsPage() {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);

  // Form state for new experiment
  const [formData, setFormData] = useState({
    name: '',
    dataset_split: 'dev',
    docs_version: 'v1',
    prompt_a: 'v1',
    model_a: 'mock',
    prompt_b: 'v2',
    model_b: 'mock',
  });

  useEffect(() => {
    loadExperiments();
  }, []);

  async function loadExperiments() {
    try {
      const res = await api.listExperiments();
      setExperiments(res.experiments);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function runExperiment() {
    setRunning(true);
    try {
      await api.runExperiment({
        name: formData.name || `Exp_${Date.now()}`,
        dataset_split: formData.dataset_split,
        docs_version: formData.docs_version,
        config_a: { prompt_version: formData.prompt_a, model_version: formData.model_a },
        config_b: { prompt_version: formData.prompt_b, model_version: formData.model_b },
      });
      loadExperiments();
    } catch (err) {
      console.error(err);
    } finally {
      setRunning(false);
    }
  }

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
        {/* New Experiment Form */}
        <div className="card">
          <h2>Run A/B Experiment</h2>
          <div className="grid grid-4" style={{ marginTop: 16, gap: 12 }}>
            <input
              className="input"
              placeholder="Experiment Name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
            />
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
              placeholder="Docs Version"
              value={formData.docs_version}
              onChange={(e) => setFormData({ ...formData, docs_version: e.target.value })}
            />
            <div></div>
          </div>

          <div className="grid grid-2" style={{ marginTop: 16, gap: 16 }}>
            <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
              <h3 style={{ marginBottom: 12 }}>Config A</h3>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="input"
                  placeholder="Prompt Version"
                  value={formData.prompt_a}
                  onChange={(e) => setFormData({ ...formData, prompt_a: e.target.value })}
                  style={{ flex: 1 }}
                />
                <input
                  className="input"
                  placeholder="Model Version"
                  value={formData.model_a}
                  onChange={(e) => setFormData({ ...formData, model_a: e.target.value })}
                  style={{ flex: 1 }}
                />
              </div>
            </div>
            <div style={{ padding: 16, background: 'var(--bg-tertiary)', borderRadius: 8 }}>
              <h3 style={{ marginBottom: 12 }}>Config B</h3>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="input"
                  placeholder="Prompt Version"
                  value={formData.prompt_b}
                  onChange={(e) => setFormData({ ...formData, prompt_b: e.target.value })}
                  style={{ flex: 1 }}
                />
                <input
                  className="input"
                  placeholder="Model Version"
                  value={formData.model_b}
                  onChange={(e) => setFormData({ ...formData, model_b: e.target.value })}
                  style={{ flex: 1 }}
                />
              </div>
            </div>
          </div>

          <button
            className="btn btn-primary"
            style={{ marginTop: 16 }}
            onClick={runExperiment}
            disabled={running}
          >
            {running ? 'Running...' : 'Run Experiment'}
          </button>
        </div>

        {/* Experiment List */}
        <div className="card" style={{ marginTop: 16 }}>
          <h2>Experiments</h2>
          {loading ? (
            <div className="loading">Loading...</div>
          ) : (
            <table className="table" style={{ marginTop: 16 }}>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Split</th>
                  <th>Config A</th>
                  <th>Config B</th>
                  <th>Status</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {experiments.map((exp) => (
                  <tr key={exp.id}>
                    <td>
                      <Link href={`/experiments/${exp.id}`}>{exp.name}</Link>
                    </td>
                    <td><span className="badge badge-info">{exp.dataset_split}</span></td>
                    <td>{exp.config_a.prompt_version}/{exp.config_a.model_version}</td>
                    <td>{exp.config_b.prompt_version}/{exp.config_b.model_version}</td>
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
                    <td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-secondary)' }}>
                      No experiments yet. Run one above!
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          )}
        </div>
      </main>
    </>
  );
}
