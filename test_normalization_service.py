from app.services.claim_normalization import normalize_claim


tests = [
    "زمین چپٹی ہے۔",
    "Zameen chapti nahi hai.",
    "Smoking sehat ke liye nuqsan deh nahi hai.",
    "Vitamin D ki kami shayad haddiyon ko kamzor kar sakti hai.",
]


for index, claim in enumerate(tests, start=1):
    print("\n" + "=" * 60)
    print(f"Test {index}")
    print("Original:", claim)

    try:
        result = normalize_claim(claim)
        print("English:", result["english_claim"])
        print("Meaning preserved:", result["meaning_preserved"])

    except Exception as exc:
        print("Normalization failed:", str(exc))