/** Taxonomy labels — must match backend TaxonomyLabel enum. */
export const TAXONOMY_LABELS = [
  'billing_seats',
  'billing_refund',
  'workspace_access',
  'permission_sharing',
  'login_sso',
  'import_export_sync',
  'bug_report',
  'feature_request',
] as const;

export type TaxonomyLabel = (typeof TAXONOMY_LABELS)[number];

/** Score dimensions used in rubric (1-5 scale). */
export const SCORE_DIMENSIONS = [
  'understanding',
  'info_strategy',
  'actionability',
  'communication',
] as const;

export type ScoreDimension = (typeof SCORE_DIMENSIONS)[number];
