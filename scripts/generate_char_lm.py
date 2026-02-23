import os
import torch
import torch.nn.functional as F
from torch_impl.language_model import TransformerLM

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

def decode(token_ids, itos):
    return "".join([itos[i] for i in token_ids])

@torch.no_grad()
def generate(model, start_text, stoi, itos, max_new_chars=200, temperature=0.6):
    model.eval()

    tokens = encode(start_text, stoi).tolist()

    for _ in range(max_new_chars):
        x = torch.tensor(tokens, dtype=torch.long)

        logits = model(x)              # (T, vocab_size)
        last_logits = logits[-1]       # (vocab_size,)

        probs = F.softmax(last_logits / temperature, dim=0)
        probs = top_k_filter(probs, k=5)
        next_id = torch.multinomial(probs, num_samples=1).item()

        tokens.append(next_id)

    return decode(tokens, itos)

def top_k_filter(probs, k=5):
    values, indices = torch.topk(probs, k)
    filtered = torch.zeros_like(probs)
    filtered[indices] = values
    filtered = filtered / filtered.sum()
    return filtered

def main():
    with open("data/tiny.txt", "r", encoding="utf-8") as f:
        text = f.read()

    stoi, itos = build_vocab(text)
    vocab_size = len(stoi)

    d = 64
    num_layers = 2

    model = TransformerLM(vocab_size, d, num_layers)

    ckpt_path = "checkpoints/char_lm.pt"
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print("Loaded checkpoint:", ckpt_path)

    out = generate(model, start_text="h", stoi=stoi, itos=itos, max_new_chars=300, temperature=1)
    print("\n--- GENERATED TEXT ---")
    print(out)

if __name__ == "__main__":
    main()