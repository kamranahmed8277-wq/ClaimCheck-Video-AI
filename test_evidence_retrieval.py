
from pprint import pprint
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s | %(name)s | %(message)s",
)

from app.services.evidence_retrieval import (
    retrieve_evidence_for_claim,
)

TEST_CLAIMS = [
    {
        "text": "If you have dark circles under your eyes, your sleep is insufficient.",
        "search_queries": [
            "dark circles under eyes lack of sleep",
            "sleep deprivation dark circles causes"
        ]
    },
    {
        "text": "Mouth ulcers may indicate folate deficiency.",
        "search_queries": [
            "mouth ulcers folate deficiency",
            "can folate deficiency cause mouth ulcers"
        ]
    },
    {
        "text": "Artificial intelligence models can generate inaccurate information.",
        "search_queries": [
            "AI models generate inaccurate information",
            "large language model hallucinations research"
        ]
    },
    {
        "text": "The Battle of Waterloo took place in 1815.",
        "search_queries": [
            "Battle of Waterloo date 1815",
            "Waterloo battle historical date"
        ]
    },
{
    "text": "Mouth ulcers may indicate folate deficiency.",
    "search_queries": [
        "folate deficiency mouth sores Cleveland Clinic",
        "folate deficiency oral ulcers medical research"
    ]
},
{
    "text": "Artificial intelligence models can generate inaccurate information.",
    "search_queries": [
        "large language model hallucinations",
        "AI systems generate false information research"
    ]
},
{
    "text": "The Battle of Waterloo took place in 1815.",
    "search_queries": [
        "Battle of Waterloo 1815 Britannica",
        "Battle of Waterloo date historical source"
    ]
},

]

for index, claim in enumerate(TEST_CLAIMS, start=1):
    print("\n" + "=" * 75)
    print(f"TEST {index}: {claim['text']}")

    result = retrieve_evidence_for_claim(claim)

    print("Retrieval status:", result["retrieval_status"])
    print("Sources found:", len(result["evidence"]))

    for source in result["evidence"]:
        pprint({
            "title": source["title"],
            "url": source["url"],
            "snippet": source["snippet"][:250],
            "search_context": source["search_context"],
        })
