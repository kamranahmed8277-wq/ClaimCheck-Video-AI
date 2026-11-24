
import logging
from pprint import pprint

from app.services.evidence_retrieval import retrieve_evidence_for_claim
from app.services.source_verification import verify_evidence

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

TEST_CLAIMS = [
    {
        "text": "Mouth ulcers may indicate folate deficiency.",
        "search_queries": [
            "folate deficiency mouth sores Cleveland Clinic",
            "folate deficiency oral ulcers medical research",
        ],
    },
    {
        "text": "Artificial intelligence models can generate inaccurate information.",
        "search_queries": [
            "AI models generate inaccurate information",
            "large language model hallucinations research",
        ],
    },
    {
        "text": "The Battle of Waterloo took place in 1815.",
        "search_queries": [
            "Battle of Waterloo date 1815",
            "Battle of Waterloo 1815 Britannica",
        ],
    },
]

for index, claim in enumerate(TEST_CLAIMS, start=1):
    print("\n" + "=" * 80)
    print(f"CLAIM {index}: {claim['text']}")

    retrieval = retrieve_evidence_for_claim(claim)
    candidates = retrieval.get("evidence", [])

    print("\nRetrieval status:", retrieval.get("retrieval_status"))
    print("Candidates found:", len(candidates))

    if not candidates:
        print("No candidates available for source verification.")
        continue

    for source_index, evidence in enumerate(candidates, start=1):
        print("\n" + "-" * 70)
        print(f"SOURCE {source_index}: {evidence.get('url')}")

        try:
            verification = verify_evidence(
                claim_text=claim["text"],
                evidence=evidence,
            )

            print("Verification result:")
            pprint(verification)

        except Exception:
            logging.exception(
                "Verification failed for %s",
                evidence.get("url"),
            )
