"""Domain taxonomy definitions for SaaS collaboration tools."""
from enum import Enum
from typing import Dict, List


class TaxonomyLabel(str, Enum):
    """Classification labels for customer service tickets."""
    BILLING_SEATS = "billing_seats"
    BILLING_REFUND = "billing_refund"
    WORKSPACE_ACCESS = "workspace_access"
    PERMISSION_SHARING = "permission_sharing"
    LOGIN_SSO = "login_sso"
    IMPORT_EXPORT_SYNC = "import_export_sync"
    BUG_REPORT = "bug_report"
    FEATURE_REQUEST = "feature_request"


class FailureTag(str, Enum):
    """Failure tags for evaluation."""
    INTENT_MISS = "intent_miss"
    MISSING_SLOT = "missing_slot"
    NO_NEXT_STEP = "no_next_step"
    POLICY_PII = "policy_pii"
    OVERCLAIM = "overclaim"
    ESCALATION_NEEDED = "escalation_needed"
    TOOL_NEEDED = "tool_needed"
    TONE_ISSUE = "tone_issue"
    CONTRADICTION = "contradiction"
    SSO_ADMIN_REQUIRED = "sso_admin_required"
    PERMISSION_MODEL_MISMATCH = "permission_model_mismatch"
    BILLING_CONTEXT_MISSING = "billing_context_missing"


# Required slots per taxonomy label
REQUIRED_SLOTS: Dict[TaxonomyLabel, List[str]] = {
    TaxonomyLabel.BILLING_SEATS: [
        "current_plan", "seat_count", "billing_cycle", "receipt_available"
    ],
    TaxonomyLabel.BILLING_REFUND: [
        "plan_type", "billing_date", "refund_amount", "payment_method"
    ],
    TaxonomyLabel.WORKSPACE_ACCESS: [
        "workspace_id", "user_role", "access_type", "error_message"
    ],
    TaxonomyLabel.PERMISSION_SHARING: [
        "resource_type", "current_permission", "target_permission", "user_count"
    ],
    TaxonomyLabel.LOGIN_SSO: [
        "idp_provider", "error_code", "is_admin", "domain_verified"
    ],
    TaxonomyLabel.IMPORT_EXPORT_SYNC: [
        "source_format", "target_format", "data_size", "sync_direction"
    ],
    TaxonomyLabel.BUG_REPORT: [
        "browser_os", "reproduction_steps", "frequency", "error_screenshot"
    ],
    TaxonomyLabel.FEATURE_REQUEST: [
        "feature_category", "use_case", "priority_indication", "alternatives_tried"
    ],
}


# Label descriptions for prompts
LABEL_DESCRIPTIONS: Dict[TaxonomyLabel, str] = {
    TaxonomyLabel.BILLING_SEATS: "Issues related to seat-based pricing, adding/removing seats, or seat allocation",
    TaxonomyLabel.BILLING_REFUND: "Refund requests, billing disputes, or payment reversal inquiries",
    TaxonomyLabel.WORKSPACE_ACCESS: "Problems accessing workspaces, workspace invitations, or workspace visibility",
    TaxonomyLabel.PERMISSION_SHARING: "Permission settings, sharing controls, or access level modifications",
    TaxonomyLabel.LOGIN_SSO: "Login issues, SSO configuration, SAML/OAuth problems, or authentication errors",
    TaxonomyLabel.IMPORT_EXPORT_SYNC: "Data import/export, third-party sync, or data migration issues",
    TaxonomyLabel.BUG_REPORT: "Technical bugs, unexpected behavior, or system errors",
    TaxonomyLabel.FEATURE_REQUEST: "Feature suggestions, enhancement requests, or capability inquiries",
}
