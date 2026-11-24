
from transformers import AutoConfig

MODEL_NAME = "cross-encoder/nli-deberta-v3-small"

print("Loading NLI model configuration...")

config = AutoConfig.from_pretrained(MODEL_NAME)

print("\nModel:", MODEL_NAME)
print("Number of labels:", config.num_labels)
print("Label mapping:", config.id2label)

print("\nNLI configuration check completed.")
