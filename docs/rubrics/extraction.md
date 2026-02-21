---
doc_id: rubric_extraction
title: "Extraction Task Evaluation Guide"
version: v1
tags: [extraction]
category: rubric
---

# Extraction Task Evaluation Guide

## What to Look For

- Correct identification of target fields or entities
- Accurate extraction of values from source text
- Proper formatting of extracted data
- Source reference for each extracted item

## Common Failure Patterns

- **Hallucination**: Extracting information not present in the source
- **Incomplete Answer**: Missing entities or fields that should be extracted
- **Format Violation**: Wrong output format (e.g. JSON when table requested)
- **Partial Answer**: Only extracting some of the requested fields

## Scoring Guidelines

### Instruction Following
- 5: All requested fields extracted in the correct format
- 3: Most fields extracted but format issues
- 1: Wrong fields extracted or format completely wrong

### Reasoning Quality
- 5: Accurate extraction with correct disambiguation of ambiguous references
- 3: Mostly accurate with minor errors
- 1: Major extraction errors or hallucinated values

### Completeness
- 5: Every instance and field extracted, nothing missed
- 3: Main items extracted but some omissions
- 1: Severely incomplete extraction

### Clarity
- 5: Clean, well-formatted output easy to consume
- 3: Correct values but messy presentation
- 1: Output is hard to parse or interpret
