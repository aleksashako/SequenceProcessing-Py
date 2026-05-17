import torch
import random
from transformers import AutoTokenizer

# -----------------------------
# Load tokenizer
# -----------------------------

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "Türkiye Cumhuriyeti'nin başkenti Ankara'dır."

# -----------------------------
# Tokenization
# -----------------------------

inputs = tokenizer(text, return_tensors="pt")
input_ids = inputs["input_ids"][0]

labels = input_ids.clone()

# -----------------------------
# MLM masking parameter (as in the paper)
# -----------------------------

mlm_probability = 0.15

# ignore special tokens
special_tokens_mask = tokenizer.get_special_tokens_mask(
    input_ids, already_has_special_tokens=True
)

special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)

# -----------------------------
# Select masking positions
# -----------------------------

probability_matrix = torch.full(labels.shape, mlm_probability)
probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

masked_indices = torch.bernoulli(probability_matrix).bool()

labels[~masked_indices] = -100  # only masked tokens contribute to loss

# -----------------------------
# 80/10/10 strategy
# -----------------------------

for i in range(len(input_ids)):

    if not masked_indices[i]:
        continue

    rand = random.random()

    # 80% MASK
    if rand < 0.8:
        input_ids[i] = tokenizer.mask_token_id

    # 10% random token
    elif rand < 0.9:
        input_ids[i] = random.randint(0, tokenizer.vocab_size - 1)

    # 10% unchanged
    else:
        pass

# -----------------------------
# Output results
# -----------------------------

print("Original Tokens:")
print(tokenizer.convert_ids_to_tokens(labels[labels != -100]))

print("\nMasked Input:")
print(tokenizer.convert_ids_to_tokens(input_ids))
