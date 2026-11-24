
from pprint import pprint

from app.services.claim_analysis import analyze_segments


test_segments = [
    {
        "start": 0.0,
        "end": 5.0,
        "text": "خشک جلد جسم میں پانی کی کمی کی وجہ سے ہو سکتی ہے۔"
    },
    {
        "start": 5.0,
        "end": 10.0,
        "text": "اس لیے پانی کا استعمال زیادہ کریں۔"
    },
    {
        "start": 10.0,
        "end": 15.0,
        "text": "گوند کتیرا دانوں کا علاج کرتا ہے۔"
    },
    {
        "start": 15.0,
        "end": 20.0,
        "text": "اگر چہرے پر دانے ہوں تو گوند کتیرا استعمال کریں۔"
    },
    {
        "start": 20.0,
        "end": 25.0,
        "text": "Pimples hain to gond katira use karein."
    },
    {
        "start": 25.0,
        "end": 30.0,
        "text": "Gond katira pimples ka ilaj karta hai."
    },
]

results = analyze_segments(test_segments)

for item in results:
    print("\n" + "=" * 70)
    print("Timestamp:", item["start"], "-", item["end"])
    print("Original:", item["text"])
    print("Category:", item["category"])
    print("Claims:")
    pprint(item["claims"])
    print("Classification failed:", item["_classification_failed"])
    print("Extraction failed:", item["_claim_extraction_failed"])
