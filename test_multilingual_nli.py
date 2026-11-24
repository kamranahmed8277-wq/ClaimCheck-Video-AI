import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"

print("Loading multilingual NLI model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("\nModel labels:")
print(model.config.id2label)


def classify(claim, evidence):
    inputs = tokenizer(
        evidence,
        claim,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    results = {
        model.config.id2label[index]: float(score)
        for index, score in enumerate(probabilities)
    }

    predicted_label = max(
        results,
        key=results.get,
    )

    return {
        "label": predicted_label,
        "probabilities": results,
    }


tests = [
    # Urdu
    {
        "name": "Urdu - Supporting",
        "claim": "سگریٹ نوشی پھیپھڑوں کے کینسر کا خطرہ بڑھاتی ہے۔",
        "evidence": "Smoking increases the risk of lung cancer.",
        "expected": "entailment",
    },
    {
        "name": "Urdu - Refuting",
        "claim": "زمین چپٹی ہے۔",
        "evidence": "The Earth is approximately spherical.",
        "expected": "contradiction",
    },
    {
        "name": "Urdu - Neutral",
        "claim": "وٹامن بی 12 کی کمی ہونٹ پھٹنے کا سبب بنتی ہے۔",
        "evidence": "The capital of France is Paris.",
        "expected": "neutral",
    },

    # English
    {
        "name": "English - Supporting",
        "claim": "Smoking increases the risk of lung cancer.",
        "evidence": "Smoking is a major risk factor for lung cancer.",
        "expected": "entailment",
    },
    {
        "name": "English - Refuting",
        "claim": "The Earth is flat.",
        "evidence": "The Earth is approximately spherical.",
        "expected": "contradiction",
    },
    {
        "name": "English - Neutral",
        "claim": "Vitamin B12 deficiency causes cracked lips.",
        "evidence": "The capital of France is Paris.",
        "expected": "neutral",
    },

    # Roman Urdu
    {
        "name": "Roman Urdu - Supporting",
        "claim": "Smoking phephron ke cancer ka khatra barhati hai.",
        "evidence": "Smoking increases the risk of lung cancer.",
        "expected": "entailment",
    },
    {
        "name": "Roman Urdu - Refuting",
        "claim": "Zameen chapti hai.",
        "evidence": "The Earth is approximately spherical.",
        "expected": "contradiction",
    },
    {
        "name": "Roman Urdu - Neutral",
        "claim": "Vitamin B12 ki kami phate honton ka sabab banti hai.",
        "evidence": "The capital of France is Paris.",
        "expected": "neutral",
    },
    {
    "name": "Roman Urdu - Refuting 2",
    "claim": "Zameen gol hai.",
    "evidence": "The Earth is flat.",
    "expected": "contradiction",
    },
    {
    "name": "Roman Urdu - Refuting 3",
    "claim": "Insan ko oxygen ki zaroorat nahi hoti.",
    "evidence": "Humans need oxygen to survive.",
    "expected": "contradiction",
    },
    {
    "name": "Roman Urdu - Refuting 4",
    "claim": "Smoking sehat ke liye nuqsan deh nahi hai.",
    "evidence": "Smoking is harmful to human health.",
    "expected": "contradiction",
    },

]

passed = 0

for test in tests:
    result = classify(
        test["claim"],
        test["evidence"],
    )

    actual = result["label"]
    expected = test["expected"]

    success = actual == expected

    if success:
        passed += 1

    print("\n" + "=" * 60)
    print("Test:", test["name"])
    print("Expected:", expected)
    print("Predicted:", actual)
    print("Confidence:", round(
        max(result["probabilities"].values()), 4
    ))
    print("Probabilities:", result["probabilities"])
    print("Result:", "PASS" if success else "FAIL")

print("\n" + "=" * 60)
print(f"TOTAL: {passed}/{len(tests)} tests passed")