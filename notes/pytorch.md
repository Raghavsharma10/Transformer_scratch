# Pytorch Intro

PyTorch = Tensors + Autogradient + nn.Module
	•	It does not change model logic
	•	It automates:
	•	fast computation
	•	gradient calculation
	•	parameter tracking

# Tensor Vs List

X = [[1.0, 2.0], [3.0, 4.0]]  --> List 

import torch
X = torch.tensor([[1.0, 2.0], [3.0, 4.0]]) --> Pytorch

tensors have .shape, .dtype
	•	support fast ops: +, @, .T
	•	can move to GPU later

# nn.Module

-> Any model or layer must inherit from nn.Module.

ex : import torch.nn as nn

     class MyLayer(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, X):
            return X

•	Always call super().__init__()
•	Put learnable components in __init__
•	Put computation in forward


# Calling a model

out = model(X)  

-> This internally calls model.forward(X)
-> Do not call .forward() directly in training code; Inside models, calling submodules’ forward is fine


# Parameters

for trainable parameters
-> self.W = nn.Parameter(torch.randn(d, d)) -> correct way


self.W = torch.randn(d, d)   # ❌ optimizer ignores this

--> Anything you want to learn must be nn.Parameter or inside nn.Linear, nn.LayerNorm, etc.


# nn.Linear 

y = x @ W + b   --> python


self.linear = nn.Linear(in_dim, out_dim)    --> pytorch
y = self.linear(x)

--> This replaces:
	•	weight matrix
	•	bias
	•	manual loops


# List of Layers -> nn.ModuleList

self.layers = [EncoderBlock(d) for _ in range(L)]  ❌ Wrong (parameters not tracked)

self.layers = nn.ModuleList(                ✅ Correct
    [EncoderBlock(d) for _ in range(L)]
)


Golden rules : 

    •	nn.Module = state + computation
	•	__init__ defines structure
	•	forward defines logic
	•	nn.Parameter = learnable
	•	ModuleList for stacking layers
	•	PyTorch mirrors math, it doesn’t invent it

# input -> embeddings file (embeddings.py)

	•	Goal of file: Convert token_ids of shape (T, ) into Transformer input matrix X ∈ ℝ^(T × d) by adding meaning (token embedding) + order (positional encoding).
	•	Imports used:
	•	torch → tensors + ops
	•	torch.nn as nn → neural network layers (nn.Module, nn.Embedding)
	•	math → constants like sqrt() and log()
	•	Class: TokenEmbedding(nn.Module)
	•	Purpose: Learnable mapping from token IDs → dense vectors
	•	Init:
	•	self.embedding = nn.Embedding(vocab_size, d) creates a trainable matrix E ∈ ℝ^(vocab_size × d)
	•	self.d = d stored for scaling
	•	Forward:
	•	Input: token_ids shape (T, )
	•	Output: self.embedding(token_ids) shape (T, d)
	•	Returns: self.embedding(token_ids) * math.sqrt(self.d) (scaling from the Transformer paper for stability)
	•	PyTorch rule: self.embedding(token_ids) is shorthand for self.embedding.forward(token_ids)
	•	Class: PositionalEncoding(nn.Module)
	•	Purpose: Inject position/order info because attention alone doesn’t know token order
	•	Init builds fixed table:
	•	pe = torch.zeros(max_len, d) → positional table shape (max_len, d)
	•	position = torch.arange(0, max_len).unsqueeze(1) → shape (max_len, 1)
	•	div_term = torch.exp(torch.arange(0, d, 2) * (-math.log(10000.0) / d)) → controls frequencies
	•	pe[:, 0::2] = sin(position * div_term) → even dims get sin
	•	pe[:, 1::2] = cos(position * div_term) → odd dims get cos
	•	self.register_buffer("pe", pe) → stored with model, moves to GPU, but not trainable
	•	Forward:
	•	Input: X shape (T, d)
	•	T = X.size(0) picks current sequence length
	•	Output: X + self.pe[:T] shape remains (T, d)
	•	Class: InputEmbedding(nn.Module)
	•	Purpose: Combine token embedding + positional encoding into one clean module
	•	Init:
	•	self.token_embedding = TokenEmbedding(vocab_size, d)
	•	self.positional_encoding = PositionalEncoding(d, max_len)
	•	Forward:
	•	Input: token_ids shape (T, )
	•	X = self.token_embedding(token_ids) → (T, d)
	•	X = self.positional_encoding(X) → (T, d)
	•	Returns final Transformer-ready X ∈ ℝ^(T × d)


# init vs forward

 Put in __init__ when it is:

✅ model configuration (fixed once)
	•	vocab_size
	•	d
	•	num_layers
	•	max_len

These define what the module is.

Put in forward when it is:

✅ runtime data (changes every call)
	•	token_ids
	•	X
	•	mask

These define what the module processes.


# test_torch_lm_loss.py

-> After we have created the transformer pipeline ; token_ids -> embeddings -> encoder -> output -> logits

Now we want the model to predict the output -

If the input is [a,b,c,d]

We want the model to predict ;
-> b at position 0
-> c at position 1
-> d at position 2

For input [a,b,c,d]
output expected is [b,c,d]

# This is called shifted language modeling


# CrossEntropyLoss expects:
	•	logits: shape (N, C)
	•	targets: shape (N,)

Where:
	•	N = number of predictions
	•	C = number of classes

In our case:
	•	N = T-1
	•	C = vocab_size

So:
	•	logits is (T-1, vocab_size)
	•	y is (T-1,)

	PyTorch does softmax internally

nn.CrossEntropyLoss() is equivalent to:

CrossEntropy(z, y) = NLLLoss(log({softmax}(z)), y)




