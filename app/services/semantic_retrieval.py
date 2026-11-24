
import re
from typing import Any

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer


class HybridEvidenceRetriever:
    """
    Retrieves evidence using semantic vector search and BM25,
    then reranks candidates with a cross-encoder.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        # Semantic embedding model
        self.model = SentenceTransformer(model_name)

        # Cross-encoder reranking model
        self.reranker = CrossEncoder(reranker_model)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def retrieve(
        self,
        claim: str,
        passages: list[str],
        top_k: int = 5,
        candidate_k: int = 20,
    ) -> list[dict[str, Any]]:

        if not claim.strip() or not passages:
            return []

        passages = [
            passage.strip()
            for passage in passages
            if passage and passage.strip()
        ]

        if not passages:
            return []

        # 1. Generate semantic embeddings.
        passage_vectors = self.model.encode(
            passages,
            normalize_embeddings=True,
        )

        query_vector = self.model.encode(
            claim,
            normalize_embeddings=True,
        ).tolist()

        # 2. Create a temporary local Qdrant collection.
        client = QdrantClient(":memory:")
        collection_name = "evidence_passages"

        vector_size = len(query_vector)

        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        # 3. Store passage embeddings in Qdrant.
        client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=index,
                    vector=vector.tolist(),
                    payload={"text": passage},
                )
                for index, (passage, vector)
                in enumerate(zip(passages, passage_vectors))
            ],
        )

        # 4. Retrieve semantic rankings.
        semantic_results = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=len(passages),
            with_payload=True,
        ).points

        semantic_ranks = {
            result.id: rank
            for rank, result in enumerate(semantic_results, start=1)
        }

        semantic_scores = {
            result.id: result.score
            for result in semantic_results
        }

        # 5. Retrieve keyword rankings with BM25.
        tokenized_passages = [
            self._tokenize(passage)
            for passage in passages
        ]

        bm25 = BM25Okapi(tokenized_passages)
        bm25_scores = bm25.get_scores(self._tokenize(claim))

        bm25_order = sorted(
            range(len(passages)),
            key=lambda index: bm25_scores[index],
            reverse=True,
        )

        bm25_ranks = {
            index: rank
            for rank, index in enumerate(bm25_order, start=1)
        }

        # 6. Combine rankings using Reciprocal Rank Fusion.
        rrf_k = 60
        combined_results = []

        for index, passage in enumerate(passages):
            semantic_rank = semantic_ranks.get(index)
            bm25_rank = bm25_ranks.get(index)

            hybrid_score = 0.0

            if semantic_rank is not None:
                hybrid_score += 1 / (rrf_k + semantic_rank)

            if bm25_scores[index] > 0:
                hybrid_score += 1 / (rrf_k + bm25_rank)

            combined_results.append({
                "text": passage,
                "semantic_score": float(
                    semantic_scores.get(index, 0.0)
                ),
                "bm25_score": float(bm25_scores[index]),
                "hybrid_score": hybrid_score,
            })

        client.close()

        # 7. Select the top hybrid candidates for reranking.
        hybrid_results = sorted(
            combined_results,
            key=lambda item: item["hybrid_score"],
            reverse=True,
        )

        candidate_count = min(
            max(candidate_k, top_k),
            len(hybrid_results),
        )

        candidates = hybrid_results[:candidate_count]

        for rank, item in enumerate(candidates, start=1):
            item["hybrid_rank"] = rank

        # 8. Score claim-passage pairs with the cross-encoder.
        pairs = [
            [claim, item["text"]]
            for item in candidates
        ]

        reranking_scores = self.reranker.predict(
            pairs,
            show_progress_bar=False,
        )

        for item, score in zip(candidates, reranking_scores):
            item["cross_encoder_score"] = float(score)

        # 9. Sort by cross-encoder relevance score.
        reranked_results = sorted(
            candidates,
            key=lambda item: item["cross_encoder_score"],
            reverse=True,
        )

        for rank, item in enumerate(reranked_results, start=1):
            item["reranked_rank"] = rank

        return reranked_results[:top_k]
