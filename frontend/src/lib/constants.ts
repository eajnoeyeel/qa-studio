/** Taxonomy labels — must match backend TaxonomyLabel enum. */
export const TAXONOMY_LABELS = [
  'reasoning',
  'math',
  'classification',
  'summarization',
  'extraction',
  'creative_writing',
  'coding',
  'open_qa',
] as const;

export type TaxonomyLabel = (typeof TAXONOMY_LABELS)[number];

/** Score dimensions used in rubric (1-5 scale). */
export const SCORE_DIMENSIONS = [
  'instruction_following',
  'reasoning_quality',
  'completeness',
  'clarity',
] as const;

export type ScoreDimension = (typeof SCORE_DIMENSIONS)[number];
