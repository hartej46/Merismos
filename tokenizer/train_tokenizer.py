import torch
from BPT_tokenizer import MinBPE 

# 1. Load the combined dataset
with open("dataset/combined_corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. Initialize and train the tokenizer
tokenizer = MinBPE()
print("Training tokenizer... (This will print progress from 0 to 1243)")
tokenizer.train(text, vocab_size=1500) 

# 3. Save the tokenizer's learned merges
tokenizer.save("trained_tokenizer_dataset/moe_tokenizer.json")

# 4. Encode the entire dataset into integers
print("Encoding dataset...")
tokens = tokenizer.encode(text)

# 5. Save as a PyTorch binary tensor for training
tensor_data = torch.tensor(tokens, dtype=torch.long)
torch.save(tensor_data, "trained_tokenizer_dataset/train.bin")
print(f"Dataset saved successfully! Total tokens: {len(tensor_data)}")