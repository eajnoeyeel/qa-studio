'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { api, EvalItem } from '@/lib/api';

export default function ItemsPage() {
  const [items, setItems] = useState<EvalItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [split, setSplit] = useState<string>('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadItems() {
      setLoading(true);
      try {
        const res = await api.listItems(page, 20, split || undefined);
        setItems(res.items);
        setTotal(res.total);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    }
    loadItems();
  }, [page, split]);

  return (
      <main className="container">
        <div className="card">
          <div className="card-header">
            <h2>Eval Items ({total})</h2>
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
                    <th>Question Preview</th>
                    <th>Response Preview</th>
                    <th>Created</th>
                  </tr>
                </thead>
                <tbody>
                  {items.map((item) => (
                    <tr key={item.id}>
                      <td>
                        <Link href={`/items/${item.id}`}>
                          {item.external_id || item.id.slice(0, 8)}
                        </Link>
                      </td>
                      <td><span className="badge badge-info">{item.split}</span></td>
                      <td style={{ maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.question.slice(0, 80)}...
                      </td>
                      <td style={{ maxWidth: 250, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                        {item.response.slice(0, 80)}...
                      </td>
                      <td>{new Date(item.created_at).toLocaleDateString()}</td>
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
                  disabled={items.length < 20}
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
