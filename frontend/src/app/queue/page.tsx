'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, HumanQueueItem } from '@/lib/api';

export default function QueuePage() {
  const [queue, setQueue] = useState<HumanQueueItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedItem, setSelectedItem] = useState<HumanQueueItem | null>(null);
  const [reviewForm, setReviewForm] = useState({
    gold_label: '',
    notes: '',
    gold_scores: {} as Record<string, number>,
  });

  useEffect(() => {
    loadQueue();
  }, []);

  async function loadQueue() {
    try {
      const items = await api.getHumanQueue(50);
      setQueue(items);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  async function submitReview() {
    if (!selectedItem) return;

    try {
      await api.submitReview({
        queue_item_id: selectedItem.id,
        evaluation_id: selectedItem.evaluation_id,
        gold_label: reviewForm.gold_label || undefined,
        gold_scores: Object.keys(reviewForm.gold_scores).length > 0 ? reviewForm.gold_scores : undefined,
        notes: reviewForm.notes || undefined,
      });
      setSelectedItem(null);
      setReviewForm({ gold_label: '', notes: '', gold_scores: {} });
      loadQueue();
    } catch (err) {
      console.error(err);
    }
  }

  return (
    <>
      <header className="header">
        <h1>CS QA Studio</h1>
        <nav className="nav">
          <Link href="/">Dashboard</Link>
          <Link href="/tickets">Tickets</Link>
          <Link href="/experiments">Experiments</Link>
          <Link href="/queue" className="active">Review Queue</Link>
        </nav>
      </header>

      <main className="container">
        <div className="card">
          <div className="card-header">
            <h2>Human Review Queue ({queue.length} pending)</h2>
            <button className="btn btn-secondary" onClick={loadQueue}>Refresh</button>
          </div>

          {loading ? (
            <div className="loading">Loading...</div>
          ) : queue.length === 0 ? (
            <div style={{ textAlign: 'center', padding: 40, color: 'var(--text-secondary)' }}>
              No items pending review
            </div>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Ticket</th>
                  <th>Reason</th>
                  <th>Priority</th>
                  <th>Created</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {queue.map((item) => (
                  <tr key={item.id}>
                    <td>
                      <Link href={`/tickets/${item.ticket_id}`}>
                        {item.ticket_id.slice(0, 8)}
                      </Link>
                    </td>
                    <td>
                      <span className={`badge badge-${
                        item.reason === 'gate_fail' ? 'error' :
                        item.reason === 'low_score' ? 'warning' :
                        'info'
                      }`}>
                        {item.reason}
                      </span>
                    </td>
                    <td>{item.priority}</td>
                    <td>{new Date(item.created_at).toLocaleString()}</td>
                    <td>
                      <button
                        className="btn btn-primary"
                        onClick={() => setSelectedItem(item)}
                      >
                        Review
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Review Modal */}
        {selectedItem && (
          <div style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(0,0,0,0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
          }}>
            <div className="card" style={{ width: '600px', maxHeight: '80vh', overflow: 'auto' }}>
              <div className="card-header">
                <h2>Submit Review</h2>
                <button
                  className="btn btn-secondary"
                  onClick={() => setSelectedItem(null)}
                >
                  Close
                </button>
              </div>

              <div style={{ marginTop: 16 }}>
                <p style={{ marginBottom: 16 }}>
                  <strong>Ticket:</strong>{' '}
                  <Link href={`/tickets/${selectedItem.ticket_id}`} target="_blank">
                    {selectedItem.ticket_id}
                  </Link>
                </p>

                <div style={{ marginBottom: 16 }}>
                  <label style={{ display: 'block', marginBottom: 8 }}>Gold Label (correct classification)</label>
                  <select
                    className="select"
                    style={{ width: '100%' }}
                    value={reviewForm.gold_label}
                    onChange={(e) => setReviewForm({ ...reviewForm, gold_label: e.target.value })}
                  >
                    <option value="">-- Select --</option>
                    <option value="billing_seats">billing_seats</option>
                    <option value="billing_refund">billing_refund</option>
                    <option value="workspace_access">workspace_access</option>
                    <option value="permission_sharing">permission_sharing</option>
                    <option value="login_sso">login_sso</option>
                    <option value="import_export_sync">import_export_sync</option>
                    <option value="bug_report">bug_report</option>
                    <option value="feature_request">feature_request</option>
                  </select>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <label style={{ display: 'block', marginBottom: 8 }}>Gold Scores (1-5)</label>
                  <div className="grid grid-2" style={{ gap: 8 }}>
                    {['understanding', 'info_strategy', 'actionability', 'communication'].map(score => (
                      <div key={score} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ flex: 1 }}>{score}</span>
                        <select
                          className="select"
                          value={reviewForm.gold_scores[score] || ''}
                          onChange={(e) => setReviewForm({
                            ...reviewForm,
                            gold_scores: {
                              ...reviewForm.gold_scores,
                              [score]: parseInt(e.target.value) || 0
                            }
                          })}
                        >
                          <option value="">--</option>
                          {[1, 2, 3, 4, 5].map(n => <option key={n} value={n}>{n}</option>)}
                        </select>
                      </div>
                    ))}
                  </div>
                </div>

                <div style={{ marginBottom: 16 }}>
                  <label style={{ display: 'block', marginBottom: 8 }}>Notes</label>
                  <textarea
                    className="input"
                    style={{ width: '100%', minHeight: 100 }}
                    value={reviewForm.notes}
                    onChange={(e) => setReviewForm({ ...reviewForm, notes: e.target.value })}
                    placeholder="Additional notes about this review..."
                  />
                </div>

                <button className="btn btn-primary" onClick={submitReview}>
                  Submit Review
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </>
  );
}
