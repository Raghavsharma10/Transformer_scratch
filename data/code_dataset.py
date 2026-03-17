import os
from tokenizer.code_tokenizer import tokenize_code, build_vocab, encode


class CodeDataset:
    """
    Loads Python files, tokenizes them, and prepares token id sequences
    for training a transformer language model.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir

        # Read all python files
        self.files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".py")
        ]

        # Tokenize all files
        self.token_sequences = []

        for file in self.files:
            with open(file, "r", encoding="utf-8") as f:
                code = f.read()

            tokens = tokenize_code(code)
            self.token_sequences.append(tokens)

        # Build vocabulary
        self.stoi, self.itos = build_vocab(self.token_sequences)

        # Encode all tokens
        self.encoded_sequences = [
            encode(tokens, self.stoi)
            for tokens in self.token_sequences
        ]

        # Flatten dataset once (cache for faster batch sampling)
        self.data = []
        for seq in self.encoded_sequences:
            self.data.extend(seq)

    def get_vocab_size(self):
        return len(self.stoi)

    def get_data(self):
        """
        Flatten all sequences into one long list of token ids.
        """
        return self.data

    def get_batch(self, T):
        """
        Sample a random training example of length T.
        Returns (x, y) where y is the next-token target.
        """
        import random

        data = self.data

        start = random.randint(0, len(data) - T - 1)

        x = data[start : start + T]
        y = data[start + 1 : start + T + 1]

        return x, y


if __name__ == "__main__":
    dataset = CodeDataset("data/code")

    print("Number of files:", len(dataset.files))
    print("Vocab size:", dataset.get_vocab_size())

    data = dataset.get_data()

    print("First 50 token ids:")
    print(data[:50])

    x, y = dataset.get_batch(16)

    print("\nSample batch (x):")
    print(x)

    print("\nSample targets (y):")
    print(y)