
from app.services.evidence_stance import EvidenceStanceClassifier

classifier = EvidenceStanceClassifier()

test_cases = [
    {
        "topic": "Health",
        "expected": "supports",
        "claim": "Smoking increases the risk of lung cancer.",
        "evidence": "Smoking is a major cause of lung cancer.",
    },
    {
        "topic": "Health",
        "expected": "refutes",
        "claim": "Smoking increases the risk of lung cancer.",
        "evidence": "Smoking does not increase the risk of lung cancer.",
    },
    {
        "topic": "Health",
        "expected": "neutral",
        "claim": "Smoking increases the risk of lung cancer.",
        "evidence": "Smoking can cause yellow stains on teeth.",
    },
    {
        "topic": "History",
        "expected": "supports",
        "claim": "The Berlin Wall fell in 1989.",
        "evidence": "The Berlin Wall was opened in November 1989.",
    },
    {
        "topic": "History",
        "expected": "refutes",
        "claim": "The Berlin Wall fell in 1989.",
        "evidence": "The Berlin Wall fell in 1991.",
    },
    {
        "topic": "History",
        "expected": "neutral",
        "claim": "The Berlin Wall fell in 1989.",
        "evidence": "Berlin is the capital of Germany.",
    },
    {
        "topic": "Technology",
        "expected": "supports",
        "claim": "Python is a programming language.",
        "evidence": "Python is a widely used high-level programming language.",
    },
    {
        "topic": "Technology",
        "expected": "refutes",
        "claim": "Python is a programming language.",
        "evidence": "Python is not a programming language.",
    },
    {
        "topic": "Technology",
        "expected": "neutral",
        "claim": "Python is a programming language.",
        "evidence": "Python is commonly used in data science.",
    },
    {
        "topic": "Finance",
        "expected": "supports",
        "claim": "Inflation reduces the purchasing power of money.",
        "evidence": "When prices rise, each unit of currency buys fewer goods and services.",
    },
    {
        "topic": "Finance",
        "expected": "refutes",
        "claim": "Inflation reduces the purchasing power of money.",
        "evidence": "Inflation increases the purchasing power of money.",
    },
    {
        "topic": "Finance",
        "expected": "neutral",
        "claim": "Inflation reduces the purchasing power of money.",
        "evidence": "Central banks publish economic reports.",
    },
]

correct = 0

print("\nNLI MULTI-TOPIC EVALUATION")
print("=" * 75)

for case in test_cases:
    result = classifier.classify(
        claim=case["claim"],
        evidence=case["evidence"],
    )

    predicted = result["stance"]
    confidence = result["confidence"]
    is_correct = predicted == case["expected"]

    if is_correct:
        correct += 1

    print(f"\nTopic: {case['topic']}")
    print(f"Expected: {case['expected']}")
    print(f"Predicted: {predicted}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Result: {'PASS' if is_correct else 'CHECK'}")

print("\n" + "=" * 75)
print(f"Correct: {correct}/{len(test_cases)}")
print(f"Accuracy: {correct / len(test_cases):.2%}")
