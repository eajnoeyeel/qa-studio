"""Domain taxonomy definitions for Q/A evaluation tasks."""
from enum import Enum
from typing import Dict, List


class TaxonomyLabel(str, Enum):
    """Classification labels for Q/A evaluation items."""
    REASONING = "reasoning"
    MATH = "math"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CREATIVE_WRITING = "creative_writing"
    CODING = "coding"
    OPEN_QA = "open_qa"


class FailureTag(str, Enum):
    """Failure tags for evaluation."""
    INSTRUCTION_MISS = "instruction_miss"
    INCOMPLETE_ANSWER = "incomplete_answer"
    HALLUCINATION = "hallucination"
    LOGIC_ERROR = "logic_error"
    FORMAT_VIOLATION = "format_violation"
    OVER_VERBOSE = "over_verbose"
    UNDER_VERBOSE = "under_verbose"
    WRONG_LANGUAGE = "wrong_language"
    UNSAFE_CONTENT = "unsafe_content"
    CITATION_MISSING = "citation_missing"
    OFF_TOPIC = "off_topic"
    PARTIAL_ANSWER = "partial_answer"


# Required slots per taxonomy label
REQUIRED_SLOTS: Dict[TaxonomyLabel, List[str]] = {
    TaxonomyLabel.REASONING: [
        "premise_identified", "conclusion_stated", "reasoning_chain", "assumptions_noted",
    ],
    TaxonomyLabel.MATH: [
        "problem_restated", "formula_used", "steps_shown", "final_answer",
    ],
    TaxonomyLabel.CLASSIFICATION: [
        "categories_listed", "chosen_category", "justification", "confidence_expressed",
    ],
    TaxonomyLabel.SUMMARIZATION: [
        "key_points", "length_appropriate", "no_new_info", "coherent_structure",
    ],
    TaxonomyLabel.EXTRACTION: [
        "target_field_identified", "value_extracted", "source_referenced", "format_correct",
    ],
    TaxonomyLabel.CREATIVE_WRITING: [
        "genre_respected", "prompt_elements_used", "tone_consistent", "narrative_structure",
    ],
    TaxonomyLabel.CODING: [
        "language_correct", "logic_sound", "edge_cases", "code_runnable",
    ],
    TaxonomyLabel.OPEN_QA: [
        "question_addressed", "factual_accuracy", "source_cited", "completeness",
    ],
}


# Label descriptions for prompts
LABEL_DESCRIPTIONS: Dict[TaxonomyLabel, str] = {
    TaxonomyLabel.REASONING: "Tasks requiring logical reasoning, inference, or multi-step deduction",
    TaxonomyLabel.MATH: "Mathematical problems including arithmetic, algebra, word problems, and proofs",
    TaxonomyLabel.CLASSIFICATION: "Tasks asking to categorize, label, or sort items into groups",
    TaxonomyLabel.SUMMARIZATION: "Tasks requiring condensing longer text into a shorter summary",
    TaxonomyLabel.EXTRACTION: "Tasks requiring extracting specific information from provided text",
    TaxonomyLabel.CREATIVE_WRITING: "Tasks involving story writing, poetry, dialogue, or creative composition",
    TaxonomyLabel.CODING: "Programming tasks including code generation, debugging, or explanation",
    TaxonomyLabel.OPEN_QA: "General knowledge questions or open-ended factual queries",
}
