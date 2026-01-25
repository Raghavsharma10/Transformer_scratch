import torch 
import torch.nn as nn
from torch.optim import Adam
from torch_impl.language_model import TransformerLM

def main():
    print("TRAINING STARTED", flush=True)
    vocab_size = 50
    d = 32
    num_layers = 2
    T = 16

    model = TransformerLM(vocab_size, d, num_layers)
    optimizer = Adam(model.parameters(), lr = 1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for step in range(1, 201):
        token_ids = torch.randint(0, vocab_size, (T, ))

        x = token_ids[:-1]
        y = token_ids[1:]

        logits = model(x)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 1 == 0:
            print(f"step {step:3d} | loss = {loss.item():.4f}")

if __name__ == "__main__":
    main()