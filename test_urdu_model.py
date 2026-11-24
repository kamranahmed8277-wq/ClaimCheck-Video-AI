
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
)

MODEL_NAME = "HuzaifaQ-12/Urdu_spelling_correction"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
model.eval()

samples = [
    "صحت اصل دولت ہے۔",
    "سیحت اصل دولت ہے۔",
    "ہمیں ساف پانی پینا چاہیے۔",
    "ہمیں صاف پانی پینا چاہیے۔",
    "ہمارے جسم کو مزبوط بنانے کے لیے متوازن خوراک ضروری ہے۔",
]

for text in samples:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
    )

    with torch.inference_mode():
        outputs = model(**inputs)

    predictions = outputs.logits.argmax(dim=-1)[0]
    tokens = tokenizer.convert_ids_to_tokens(
        inputs["input_ids"][0]
    )

    print(f"\nInput: {text}")
    print("Token predictions:")

    for token, prediction in zip(tokens, predictions.tolist()):
        if token not in ("[CLS]", "[SEP]", "[PAD]"):
            print(f"  {token} -> {model.config.id2label[prediction]}")
