from app.services.source_verification import (
    _extract_relevant_passages,
)

claim = (
    "white spots on nails zinc deficiency"
)

page_text = """
White spots on nails are common and often result from
minor injuries to the nail plate.

Some people believe that a lack of minerals, including
iron, calcium and zinc, may cause white spots on nails.

Healthcare providers and medical researchers are not
sure whether nutritional deficiencies cause white spots
to appear on your nails.

Nail changes can have several different causes.
"""

passages = _extract_relevant_passages(
    claim_text=claim,
    page_text=page_text,
    max_passages=3,
    search_context=(
        "white spots on nails zinc deficiency"
    ),
)

print("\nExtracted passages:")
print("-" * 60)

for index, passage in enumerate(passages, start=1):
    print(f"\nPassage {index}")
    print("Text:", passage["text"])
    print("Relevance:", passage["relevance_score"])
    print("Matched keywords:", passage["matched_keywords"])

print("\nTotal passages:", len(passages))