
from app.services.claim_analysis import analyze_segments


sample_segments = [
    {
        "start": 1.1,
        "end": 3.7,
        "text": "Talking about medicine side effects.",
    },
    {
        "start": 3.7,
        "end": 9.55,
        "text": "Hi doctor, I think the new medicine is causing problems.",
    },
    {
        "start": 12.89,
        "end": 19.02,
        "text": "I've been feeling really tired and my hands shake a little.",
    },
    {
        "start": 26.48,
        "end": 32.62,
        "text": "Those can be side effects.",
    },
    {
        "start": 51.19,
        "end": 53.19,
        "text": "Thanks, Doctor. I'll be careful.",
    },
]


results = analyze_segments(sample_segments)

for result in results:
    print("-" * 60)
    print(f"Timestamp: {result['start']} - {result['end']}")
    print(f"Text: {result['text']}")
    print(f"Category: {result['category']}")
    print(f"Reason: {result['reason']}")
    print(f"Needs fact-check: {result['requires_fact_check']}")
