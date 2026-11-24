
from app.services.semantic_retrieval import HybridEvidenceRetriever

retriever = HybridEvidenceRetriever()

test_cases = [
    {
        "topic": "Health",
        "claim": "Regular exercise can reduce the risk of heart disease.",
        "passages": [
            "Exercise can lower the risk of coronary heart disease.",
            "The weather will be sunny tomorrow.",
            "Physical activity supports cardiovascular health.",
            "A new film will be released next month.",
        ],
        "relevant": {
            "Exercise can lower the risk of coronary heart disease.",
            "Physical activity supports cardiovascular health.",
        },
    },
    {
        "topic": "History",
        "claim": "The printing press helped spread knowledge across Europe.",
        "passages": [
            "Johannes Gutenberg's printing technology helped books become more widely available.",
            "The modern internet connects computers around the world.",
            "Printing made it easier to distribute written information.",
            "The weather was unusually cold last winter.",
        ],
        "relevant": {
            "Johannes Gutenberg's printing technology helped books become more widely available.",
            "Printing made it easier to distribute written information.",
        },
    },
    {
        "topic": "Technology",
        "claim": "Machine learning systems can identify patterns in data.",
        "passages": [
            "Machine learning algorithms discover patterns from datasets.",
            "Football is played by two teams.",
            "Some AI models learn statistical relationships from training data.",
            "The printing press changed the distribution of books.",
        ],
        "relevant": {
            "Machine learning algorithms discover patterns from datasets.",
            "Some AI models learn statistical relationships from training data.",
        },
    },
]


def evaluate_retrieval(results, relevant_passages, k=3):
    retrieved = [item["text"] for item in results[:k]]

    relevant_retrieved = [
        passage for passage in retrieved
        if passage in relevant_passages
    ]

    precision = len(relevant_retrieved) / k
    recall = len(relevant_retrieved) / len(relevant_passages)

    reciprocal_rank = 0.0

    for rank, passage in enumerate(retrieved, start=1):
        if passage in relevant_passages:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, reciprocal_rank


for case in test_cases:
    results = retriever.retrieve(
        claim=case["claim"],
        passages=case["passages"],
        top_k=3,
    )

    precision, recall, mrr = evaluate_retrieval(
        results,
        case["relevant"],
        k=3,
    )

    print(f"\nTopic: {case['topic']}")
    print("Claim:", case["claim"])
    print("Precision@3:", round(precision, 3))
    print("Recall@3:", round(recall, 3))
    print("MRR:", round(mrr, 3))

    print("\nRetrieved passages:")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result['text']}")
