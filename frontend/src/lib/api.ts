/**
 * API client for QA Evaluation Studio backend.
 */

const API_BASE = '/api';

export interface EvalItem {
  id: string;
  external_id?: string;
  split: string;
  system_prompt?: string;
  question: string;
  response: string;
  metadata?: Record<string, unknown>;
  masked_text?: string;
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
  item_id: string;
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
  item_id: string;
  evaluation_id: string;
  reason: string;
  priority: number;
  created_at: string;
  reviewed: boolean;
}

export interface ExperimentSummary {
  experiment_id: string;
  total_items: number;
  gate_fail_rate_a: number;
  gate_fail_rate_b: number;
  top_tag_delta: Record<string, number>;
  avg_scores_a: Record<string, number>;
  avg_scores_b: Record<string, number>;
  completeness_distribution_a: Record<number, number>;
  completeness_distribution_b: Record<number, number>;
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
  // Items
  listItems: (page = 1, pageSize = 50, split?: string) => {
    const params = new URLSearchParams({ page: String(page), page_size: String(pageSize) });
    if (split) params.append('split', split);
    return fetchAPI<{ items: EvalItem[]; total: number; page: number; page_size: number }>(
      `/items?${params}`
    );
  },

  getItem: (id: string) =>
    fetchAPI<{ item: EvalItem; evaluations: Evaluation[] }>(`/items/${id}`),

  // Evaluations
  listEvaluations: (itemId: string) =>
    fetchAPI<{ evaluations: Evaluation[] }>(`/evaluations?item_id=${itemId}`),

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
