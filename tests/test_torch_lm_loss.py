import torch
import torch.nn as nn
from torch_impl.language_model import TransformerLM

def test_transformer_lm_loss_runs():
    vocab_size = 50
    d = 16
    num_layers = 2
    T = 8

    model = TransformerLM(vocab_size, d, num_layers)

    token_ids = torch.randint(0, vocab_size, (T,))

    #Input all tokens except last token
    x = token_ids[:-1]

    #target all tokens except first
    y = token_ids[1:]

    logits = model(x)           # (T-1, vocab_size)

    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, y)

    assert loss.item() > 0

    


