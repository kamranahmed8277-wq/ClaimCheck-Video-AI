
from app.services.semantic_retrieval import HybridEvidenceRetriever

claim = "Regular physical activity can improve cardiovascular health."

passages = [
    "Exercise can reduce the risk of coronary heart disease and stroke.",
    "Physical activity can improve mood, sleep quality and energy.",
    "The weather will be sunny tomorrow.",
    "Regular movement supports heart health and helps reduce the risk of disease.",
    "Adults should aim for at least 150 minutes of physical activity each week.",
]

retriever = HybridEvidenceRetriever()

results = retriever.retrieve(
    claim=claim,
    passages=passages,
    top_k=5,
)

print("\nHybrid evidence retrieval results:")

for index, result in enumerate(results, start=1):
    print(f"\nPassage {index}")
    print("Hybrid score:", round(result["hybrid_score"], 5))
    print("Semantic score:", round(result["semantic_score"], 4))
    print("BM25 score:", round(result["bm25_score"], 4))
    print("Text:", result["text"])
