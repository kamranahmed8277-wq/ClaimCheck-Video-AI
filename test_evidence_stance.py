
from app.services.evidence_stance import classify_evidence_passages


def run_test(test_name, claim, passages, expected):
    print(f"\n{'=' * 65}")
    print(f"TEST: {test_name}")
    print(f"CLAIM: {claim}")

    try:
        results = classify_evidence_passages(claim, passages)

        passed = True

        for result in results:
            index = result["passage_index"]
            stance = result["stance"]
            confidence = result["confidence"]
            reason = result["reason"]

            print(f"\nPassage {index}: {passages[index]['text']}")
            print(f"Stance: {stance}")
            print(f"Confidence: {confidence}")
            print(f"Reason: {reason}")

            if stance != expected[index]:
                passed = False

        if len(results) != len(expected):
            passed = False

        print("\nRESULT:", "PASS" if passed else "CHECK")

        return passed

    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        return False


def main():
    tests = [
        {
            "name": "Supporting evidence",
            "claim": "Smoking increases the risk of lung cancer.",
            "passages": [
                {
                    "text": (
                        "Smoking is a major risk factor for lung cancer. "
                        "People who smoke have a higher risk of developing "
                        "the disease."
                    )
                }
            ],
            "expected": ["supports"],
        },
        {
            "name": "Contradicting evidence",
            "claim": "The Earth is flat.",
            "passages": [
                {
                    "text": (
                        "Earth is approximately spherical, slightly flattened "
                        "at the poles, and not a flat surface."
                    )
                }
            ],
            "expected": ["refutes"],
        },
        {
            "name": "Neutral evidence",
            "claim": "Vitamin B12 deficiency causes cracked lips.",
            "passages": [
                {
                    "text": (
                        "Cracked lips can occur for several reasons, including "
                        "dry weather, dehydration, and irritation."
                    )
                }
            ],
            "expected": ["neutral"],
        },
        {
            "name": "Mixed passage classifications",
            "claim": "Smoking increases the risk of lung cancer.",
            "passages": [
                {
                    "text": (
                        "Smoking increases the risk of developing lung cancer."
                    )
                },
                {
                    "text": (
                        "Smoking does not increase lung cancer risk."
                    )
                },
                {
                    "text": (
                        "Smoking is a common habit in several countries."
                    )
                },
            ],
            "expected": ["supports", "refutes", "neutral"],
        },
    ]

    passed_count = 0

    for test in tests:
        if run_test(
            test["name"],
            test["claim"],
            test["passages"],
            test["expected"],
        ):
            passed_count += 1

    print("\n" + "=" * 65)
    print(f"TOTAL: {passed_count}/{len(tests)} tests passed.")
    print("=" * 65)


if __name__ == "__main__":
    main()
