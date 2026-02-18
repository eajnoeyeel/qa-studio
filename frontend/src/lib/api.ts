/**
 * API client for CS QA Studio backend.
 */

const API_BASE = '/api';

export interface Message {
  role: string;
  content: string;
  timestamp?: string;
}

export interface Ticket {
  id: string;
  external_id?: string;
  split: string;
  conversation: Message[];
  candidate_response: string;
  metadata?: Record<string, unknown>;
  normalized_text?: string;
  created_at: string;
}

export interface GateResult {
  gate_type: string;
  passed: boolean;
  reason?: string;
  evidence?: string;
}

export interface ScoreResult {
  score_type: string;
  score: number;
  justification: string;
}

export interface JudgeOutput {
  id: string;
  evaluation_id: string;
  gates: GateResult[];
  scores: ScoreResult[];
  failure_tags: string[];
  summary_of_issue: string;
  what_to_fix: string;
  rag_citations: string[];
  created_at: string;
}

export interface Evaluation {
  id: string;
  ticket_id: string;
  prompt_version: string;
  model_version: string;
  docs_version: string;
  classification?: {
    label: string;
    confidence: number;
    required_slots: string[];
    detected_slots: Record<string, string>;
    missing_slots: string[];
  };
  judge_output?: JudgeOutput;
  trace_id?: string;
  created_at: string;
}

export interface HumanQueueItem {
  id: string;
  ticket_id: string;
  evaluation_id: string;
  reason: string;
  priority: number;
  created_at: string;
  reviewed: boolean;
}

export interface ExperimentSummary {
  experiment_id: string;
  total_tickets: number;
  gate_fail_rate_a: number;
  gate_fail_rate_b: number;
  top_tag_delta: Record<string, number>;
  avg_scores_a: Record<string, number>;
  avg_scores_b: Record<string, number>;
  actionability_distribution_a: Record<number, number>;
  actionability_distribution_b: Record<number, number>;
  human_queue_count: number;
  human_queue_rate: number;
}

export interface Experiment {
  id: string;
  name: string;
  dataset_split: string;
  docs_version: string;
  config_a: { prompt_version: string; model_version: string };
  config_b: { prompt_version: string; model_version: string };
  summary?: ExperimentSummary;
  created_at: string;
  completed_at?: string;
}

async function fetchAPI<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!res.ok) {
    const error = await res.text();
    throw new Error(error || `API error: ${res.status}`);
  }

  return res.json();
}

export const api = {
  // Tickets
  listTickets: (page = 1, pageSize = 50, split?: string) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (split) params.append('split', split);
    return fetchAPI<{ tickets: Ticket[]; total: number; page: number; page_size: number }>(
      `/tickets?${params}`
    );
  },

  getTicket: (id: string) =>
    fetchAPI<{ ticket: Ticket; evaluations: Evaluation[] }>(`/tickets/${id}`),

  // Evaluations
  listEvaluations: (ticketId: string) =>
    fetchAPI<{ evaluations: Evaluation[] }>(`/evaluations?ticket_id=${ticketId}`),

  runEvaluation: (data: {
    dataset_split: string;
    prompt_version: string;
    model_version: string;
    docs_version: string;
  }) =>
    fetchAPI<{
      processed_count: number;
      gate_fail_count: number;
      human_queue_count: number;
      top_tags: Record<string, number>;
      avg_scores: Record<string, number>;
    }>('/evaluate/run', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Experiments
  listExperiments: () =>
    fetchAPI<{ experiments: Experiment[] }>('/experiments'),

  getExperiment: (id: string) =>
    fetchAPI<Experiment>(`/experiments/${id}`),

  runExperiment: (data: {
    dataset_split: string;
    docs_version: string;
    config_a: { prompt_version: string; model_version: string };
    config_b: { prompt_version: string; model_version: string };
    name?: string;
  }) =>
    fetchAPI<{ experiment_id: string; summary: ExperimentSummary }>('/experiment/ab', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Human Queue
  getHumanQueue: (limit = 50) =>
    fetchAPI<HumanQueueItem[]>(`/human/queue?limit=${limit}`),

  submitReview: (data: {
    queue_item_id: string;
    evaluation_id: string;
    gold_label?: string;
    gold_gates?: Record<string, boolean>;
    gold_scores?: Record<string, number>;
    gold_tags?: string[];
    notes?: string;
  }) =>
    fetchAPI('/human/review', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  // Reports
  getReportSummary: (split: string) =>
    fetchAPI<{
      dataset_split: string;
      total_evaluations: number;
      gate_fail_rate: number;
      avg_scores: Record<string, number>;
      tag_distribution: Record<string, number>;
      human_queue_stats: Record<string, number>;
    }>(`/reports/summary?dataset_split=${split}`),

  // Health
  health: () => fetchAPI<{ status: string; langfuse_enabled: boolean; llm_provider: string }>('/health'),
};
