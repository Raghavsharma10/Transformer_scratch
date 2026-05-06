import os
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_impl.language_model import TransformerLM

def build_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos

def encode(text, stoi):
    return torch.tensor([stoi[ch] for ch in text], dtype=torch.long)

def get_batch(data, start, T):
    x = data[start : start + T]
    y = data[start + 1 : start + T + 1]
    return x, y

def main():
    with open("data/alice.txt", "r", encoding="utf-8") as f:
        text = f.read()

    stoi, itos = build_vocab(text)
    data = encode(text, stoi)

    vocab_size = len(stoi)
    d = 128
    num_layers = 4
    T = 128

    model = TransformerLM(vocab_size, d, num_layers)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/char_lm.pt"

    if os.path.exists("checkpoints/char_lm.pt"):
        #model.load_state_dict(torch.load("checkpoints/char_lm.pt"))
        print("Loaded checkpoint: checkpoints/char_lm.pt")

    else:
        print("No checkpoint found, training from scratch.")

    optimizer = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    print("TRAINING STARTED")
    print("vocab_size =", vocab_size)

    for step in range(1, 40001):
        start = torch.randint(0, len(data) - T - 1, (1,)).item()
        x, y = get_batch(data, start, T)

        logits = model(x)         # (T, vocab_size)
        loss = loss_fn(logits, y) # (T,)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 2000 == 0:
            print(f"step {step:3d} | loss = {loss.item():.4f}")
    
    torch.save(model.state_dict(), ckpt_path)
    print("Saved checkpoint:", ckpt_path)

if __name__ == "__main__":
    main()