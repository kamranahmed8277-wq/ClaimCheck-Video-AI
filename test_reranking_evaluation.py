
from app.services.semantic_retrieval import HybridEvidenceRetriever


TEST_CASES = [
    {
        "topic": "Health",
        "claim": "Regular exercise can reduce the risk of heart disease.",
        "passages": [
            "Exercise can lower the risk of coronary heart disease.",
            "Physical activity supports cardiovascular health.",
            "The weather will be sunny tomorrow.",
            "A new film will be released next month.",
            "Regular movement can improve heart health and reduce cardiovascular risks.",
        ],
        "relevant": {
            "Exercise can lower the risk of coronary heart disease.",
            "Physical activity supports cardiovascular health.",
            "Regular movement can improve heart health and reduce cardiovascular risks.",
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
            "The printing press contributed to the spread of literacy and ideas in Europe.",
        ],
        "relevant": {
            "Johannes Gutenberg's printing technology helped books become more widely available.",
            "Printing made it easier to distribute written information.",
            "The printing press contributed to the spread of literacy and ideas in Europe.",
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
            "Data-driven models can detect useful structures in large datasets.",
        ],
        "relevant": {
            "Machine learning algorithms discover patterns from datasets.",
            "Some AI models learn statistical relationships from training data.",
            "Data-driven models can detect useful structures in large datasets.",
        },
    },
]


def calculate_metrics(ranked_passages, relevant_passages, k=3):
    top_results = ranked_passages[:k]

    relevant_count = sum(
        1
        for passage in top_results
        if passage in relevant_passages
    )

    precision = relevant_count / k
    recall = relevant_count / len(relevant_passages)

    reciprocal_rank = 0.0

    for rank, passage in enumerate(ranked_passages, start=1):
        if passage in relevant_passages:
            reciprocal_rank = 1 / rank
            break

    return precision, recall, reciprocal_rank


def evaluate_case(retriever, case):
    results = retriever.retrieve(
        claim=case["claim"],
        passages=case["passages"],
        top_k=len(case["passages"]),
        candidate_k=len(case["passages"]),
    )

    # Reconstruct the original hybrid ranking.
    hybrid_ranking = sorted(
        results,
        key=lambda item: item["hybrid_rank"],
    )

    # Reconstruct the cross-encoder ranking.
    reranked_ranking = sorted(
        results,
        key=lambda item: item["reranked_rank"],
    )

    relevant = case["relevant"]

    hybrid_texts = [
        item["text"] for item in hybrid_ranking
    ]

    reranked_texts = [
        item["text"] for item in reranked_ranking
    ]

    hybrid_metrics = calculate_metrics(
        hybrid_texts,
        relevant,
    )

    reranked_metrics = calculate_metrics(
        reranked_texts,
        relevant,
    )

    print("\n" + "=" * 65)
    print("TOPIC:", case["topic"])
    print("CLAIM:", case["claim"])

    print("\nHYBRID RETRIEVAL RANKING")

    for rank, item in enumerate(hybrid_ranking, start=1):
        print(
            f"{rank}. Hybrid score: {item['hybrid_score']:.5f} | "
            f"Text: {item['text']}"
        )

    print("\nCROSS-ENCODER RERANKING")

    for rank, item in enumerate(reranked_ranking, start=1):
        print(
            f"{rank}. Cross-encoder score: "
            f"{item['cross_encoder_score']:.4f} | "
            f"Original hybrid rank: {item['hybrid_rank']} | "
            f"Text: {item['text']}"
        )

    print("\nMETRIC COMPARISON")
    print(
        f"{'Metric':<18}"
        f"{'Hybrid':<15}"
        f"{'Reranked':<15}"
        f"{'Change':<12}"
    )

    metric_names = [
        "Precision@3",
        "Recall@3",
        "MRR",
    ]

    for name, before, after in zip(
        metric_names,
        hybrid_metrics,
        reranked_metrics,
    ):
        change = after - before

        print(
            f"{name:<18}"
            f"{before:<15.3f}"
            f"{after:<15.3f}"
            f"{change:+.3f}"
        )

    return hybrid_metrics, reranked_metrics


def main():
    print("Initializing Hybrid Evidence Retriever...")
    retriever = HybridEvidenceRetriever()

    all_hybrid_metrics = []
    all_reranked_metrics = []

    for case in TEST_CASES:
        hybrid_metrics, reranked_metrics = evaluate_case(
            retriever,
            case,
        )

        all_hybrid_metrics.append(hybrid_metrics)
        all_reranked_metrics.append(reranked_metrics)

    # Calculate average metrics across all topics.
    hybrid_averages = [
        sum(values[index] for values in all_hybrid_metrics)
        / len(all_hybrid_metrics)
        for index in range(3)
    ]

    reranked_averages = [
        sum(values[index] for values in all_reranked_metrics)
        / len(all_reranked_metrics)
        for index in range(3)
    ]

    print("\n" + "=" * 65)
    print("FINAL AVERAGE METRICS ACROSS ALL TOPICS")

    for name, before, after in zip(
        ["Precision@3", "Recall@3", "MRR"],
        hybrid_averages,
        reranked_averages,
    ):
        print(
            f"{name:<18}"
            f"Hybrid: {before:.3f} | "
            f"Reranked: {after:.3f} | "
            f"Change: {after - before:+.3f}"
        )


if __name__ == "__main__":
    main()
