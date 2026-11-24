
import re

from app.services.source_verification import verify_evidence
from app.services.semantic_retrieval import HybridEvidenceRetriever


claim = "Regular physical activity can improve cardiovascular health."

evidence = {
    "title": "Exercise: Health benefits",
    "url": "https://www.nhs.uk/live-well/exercise/exercise-health-benefits/",
    "snippet": "Regular physical activity provides important health benefits.",
    "search_context": "physical activity health benefits",
}

print("Retrieving the NHS evidence page...")

result = verify_evidence(claim, evidence)

print("Page retrieved:", result.get("page_retrieved"))
print("Page content extracted:", result.get("page_content_extracted"))

page_text = result.get("page_text", "")

if not page_text:
    raise RuntimeError("No page text was returned. Check the source verification result.")

# Create candidate sentences from the complete extracted page text.
passages = re.split(r"(?<=[.!?])\s+|\n+", page_text)

passages = [
    sentence.strip()
    for sentence in passages
    if len(sentence.strip()) >= 40
]

print("Candidate passages:", len(passages))

retriever = HybridEvidenceRetriever()

results = retriever.retrieve(
    claim=claim,
    passages=passages,
    top_k=5,
)

print("\nTop 5 cross-encoder reranked results:")

for index, item in enumerate(results, start=1):
    print(f"\nPassage {index}")
    print("Hybrid rank:", item["hybrid_rank"])
    print("Reranked rank:", item["reranked_rank"])
    print("Hybrid score:", round(item["hybrid_score"], 5))
    print("Semantic score:", round(item["semantic_score"], 4))
    print("BM25 score:", round(item["bm25_score"], 4))
    print("Cross-encoder score:", round(item["cross_encoder_score"], 4))
    print("Text:", item["text"])