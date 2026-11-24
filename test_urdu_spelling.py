
from LughaatNLP import LughaatNLP

print("Loading Urdu spelling checker...")
checker = LughaatNLP()

samples = [
    "صحت اصل دولت ہے۔",
    "سیحت اصل دولت ہے۔",
    "ہمیں صاف پانی پینا چاہیے۔",
    "ہمیں ساف پانی پینا چاہیے۔",
    "ہمارے جسم کو مزبوط بنانے کے لیے متوازن خوراک ضروری ہے۔",
    "ہمیں پھل اور سبزیاں کھانی چاہیے۔",
]

for sentence in samples:
    print("\nOriginal:", sentence)

    try:
        suggestion = checker.corrected_sentence_spelling(sentence, 80)
        print("Suggestion:", suggestion)
    except Exception as error:
        print("Error:", error)
