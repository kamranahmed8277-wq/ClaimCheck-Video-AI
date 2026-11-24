from pprint import pprint

from app.services.claim_analysis import analyze_segments


segments = [
    {
        "start": 0,
        "end": 5,
        "text": "اگر آپ کے ناخنوں کے اندر سفید رنگ کے اسپورٹس ہیں تو آپ کو زنک کی ڈیفینسی ہے۔",
    },
    {
        "start": 5,
        "end": 10,
        "text": "اپنی زنک کی کمی کو پورا کرو۔",
    },
    {
        "start": 10,
        "end": 15,
        "text": "اگر آپ کے انکھوں کے نیچے ڈارک سرکلز ہیں تو آپ کی نیند صحیح سے پوری نہیں ہو رہی۔",
    },
    {
        "start": 15,
        "end": 20,
        "text": "اپنی نیند کو پورا کرو۔",
    },
    {
        "start": 20,
        "end": 25,
        "text": "اگر آپ کے لپس پہ کریکس اتے ہیں تو آپ کو وٹامن بی 12 کی ڈیفینسی ہے۔",
    },
    {
        "start": 25,
        "end": 30,
        "text": "اگر منہ میں چھالے نکلتے ہیں تو آپ کو فولٹ کی ڈیفینسی ہو سکتی ہے۔",
    },
    {
        "start": 30,
        "end": 35,
        "text": "اگر آپ کی ائز میں پفیس ہے تو آپ کے باڈی کے اندر سالٹ بہت زیادہ ہے۔",
    },
    {
        "start": 35,
        "end": 40,
        "text": "سکن آپ کی ڈرائی رہتی ہے تو پانی کا استعمال زیادہ کرو۔",
    },
    {
        "start": 40,
        "end": 45,
        "text": "اگر آپ کے چہرے پہ دانے زیادہ نکلتے ہیں تو ایک مہینہ لگاتار آپ نے گوند کتیرا کا استعمال کرنا ہے۔",
    },
]

results = analyze_segments(segments)

for result in results:
    print("\n" + "=" * 70)
    print("Timestamp:", result["start"], "-", result["end"])
    print("Original:", result["text"])
    print("Category:", result["category"])
    print("Claims:")
    pprint(result["claims"])
    print("Classification failed:", result.get("_classification_failed"))
    print("Extraction failed:", result.get("_claim_extraction_failed"))