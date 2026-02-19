'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import { api, Ticket, Evaluation } from '@/lib/api';

export default function TicketDetailPage() {
  const params = useParams();
  const ticketId = params.id as string;

  const [ticket, setTicket] = useState<Ticket | null>(null);
  const [evaluations, setEvaluations] = useState<Evaluation[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadTicket() {
      try {
        const res = await api.getTicket(ticketId);
        setTicket(res.ticket);
        setEvaluations(res.evaluations);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load ticket');
      } finally {
        setLoading(false);
      }
    }
    loadTicket();
  }, [ticketId]);

  if (loading) return <div className="loading">Loading...</div>;
  if (error) return <div className="error">{error}</div>;
  if (!ticket) return <div className="error">Ticket not found</div>;

  const latestEval = evaluations[0];
  const judgeOutput = latestEval?.judge_output;

  return (
      <main className="container">
        <div style={{ marginBottom: 16 }}>
          <Link href="/tickets">&larr; Back to Tickets</Link>
        </div>

        <div className="grid grid-2">
          {/* Conversation */}
          <div className="card">
            <h2>Conversation</h2>
            <div className="conversation" style={{ marginTop: 16 }}>
              {ticket.conversation.map((msg, idx) => (
                <div key={idx} className={`message message-${msg.role}`}>
                  <div className="message-role">{msg.role.toUpperCase()}</div>
                  <div>{msg.content}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Candidate Response */}
          <div className="card">
            <h2>Candidate Response</h2>
            <div className="message message-assistant" style={{ marginTop: 16 }}>
              {ticket.candidate_response}
            </div>

            {latestEval?.classification && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 8 }}>Classification</h3>
                <span className="badge badge-info">{latestEval.classification.label}</span>
                <span style={{ marginLeft: 8, color: 'var(--text-secondary)' }}>
                  ({(latestEval.classification.confidence * 100).toFixed(0)}% confidence)
                </span>
                {latestEval.classification.missing_slots.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    <span style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>Missing slots: </span>
                    {latestEval.classification.missing_slots.map(slot => (
                      <span key={slot} className="tag">{slot}</span>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Evaluation Results */}
        {judgeOutput && (
          <div className="card" style={{ marginTop: 16 }}>
            <h2>Evaluation Results</h2>

            <div className="grid grid-2" style={{ marginTop: 16 }}>
              {/* Gates */}
              <div>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 12 }}>Gates</h3>
                {judgeOutput.gates.map((gate) => (
                  <div key={gate.gate_type} style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <span className={`badge ${gate.passed ? 'badge-success' : 'badge-error'}`}>
                      {gate.passed ? 'PASS' : 'FAIL'}
                    </span>
                    <span style={{ marginLeft: 8 }}>{gate.gate_type}</span>
                    {gate.reason && (
                      <span style={{ marginLeft: 8, color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
                        ({gate.reason})
                      </span>
                    )}
                  </div>
                ))}
              </div>

              {/* Scores */}
              <div>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 12 }}>Scores</h3>
                {judgeOutput.scores.map((score) => (
                  <div key={score.score_type} style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
                      <span>{score.score_type}</span>
                      <span>{score.score}/5</span>
                    </div>
                    <div className="score-bar">
                      <div className="score-bar-bg">
                        <div
                          className={`score-bar-fill score-${score.score}`}
                          style={{ width: `${(score.score / 5) * 100}%` }}
                        />
                      </div>
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', marginTop: 4 }}>
                      {score.justification}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Failure Tags */}
            {judgeOutput.failure_tags.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 8 }}>Failure Tags</h3>
                {judgeOutput.failure_tags.map(tag => (
                  <span key={tag} className="tag" style={{ background: 'rgba(239, 68, 68, 0.2)', color: 'var(--error)' }}>
                    {tag}
                  </span>
                ))}
              </div>
            )}

            {/* Summary & Fix */}
            <div className="grid grid-2" style={{ marginTop: 16 }}>
              <div>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 8 }}>Summary</h3>
                <p style={{ color: 'var(--text-secondary)' }}>{judgeOutput.summary_of_issue}</p>
              </div>
              <div>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 8 }}>What to Fix</h3>
                <p style={{ color: 'var(--text-secondary)' }}>{judgeOutput.what_to_fix}</p>
              </div>
            </div>

            {/* RAG Citations */}
            {judgeOutput.rag_citations.length > 0 && (
              <div style={{ marginTop: 16 }}>
                <h3 style={{ fontSize: '0.875rem', marginBottom: 8 }}>Referenced Documents</h3>
                {judgeOutput.rag_citations.map(docId => (
                  <span key={docId} className="tag">{docId}</span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Evaluation History */}
        {evaluations.length > 1 && (
          <div className="card" style={{ marginTop: 16 }}>
            <h2>Evaluation History</h2>
            <table className="table" style={{ marginTop: 16 }}>
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Prompt Version</th>
                  <th>Model Version</th>
                  <th>Docs Version</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {evaluations.map((evalItem) => (
                  <tr key={evalItem.id}>
                    <td>{evalItem.id.slice(0, 8)}</td>
                    <td>{evalItem.prompt_version}</td>
                    <td>{evalItem.model_version}</td>
                    <td>{evalItem.docs_version}</td>
                    <td>{new Date(evalItem.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </main>
  );
}
