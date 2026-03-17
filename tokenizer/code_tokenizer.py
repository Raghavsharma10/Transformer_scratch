import tokenize 
from io import BytesIO

SPECIAL_TOKENS = ["<PAD>", "<BOS>", "<EOS>", "<UNK>", "<INDENT>", "<DEDENT>", "<NEWLINE>"]

def tokenize_code(code):
    """
    Convert Python code to code tokens
    """

    tokens = []

    token_stream = tokenize.tokenize(BytesIO(code.encode("utf-8")).readline)

    for tok in token_stream:
        if tok.type == tokenize.ENCODING:
            continue
        if tok.type == tokenize.ENDMARKER:
            continue

        if tok.type == tokenize.INDENT:
            tokens.append("<INDENT>")
            continue

        if tok.type == tokenize.DEDENT:
            tokens.append("<DEDENT>")
            continue

        if tok.type == tokenize.NEWLINE:
            tokens.append("<NEWLINE>")
            continue

        tokens.append(tok.string)
    
    return tokens


def build_vocab(token_sequences):
    """
    Build vocabulary from tokenized code.
    """
    vocab = set()

    for seq in token_sequences:
        vocab.update(seq)

    vocab = SPECIAL_TOKENS + sorted(list(vocab - set(SPECIAL_TOKENS)))

    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}

    return stoi, itos

def encode(tokens, stoi):
    """
    Convert tokens into token ids.
    """
    return [stoi[token] for token in tokens]

def decode(ids, itos):
    """
    Convert token ids back to tokens.
    """
    return [itos[i] for i in ids]


if __name__ == "__main__":
    sample_code = """
for i in range(n):
    print(i)
"""

    print("Original code:")
    print(sample_code)

    tokens = tokenize_code(sample_code)
    print("\nTokens:")
    print(tokens)

    stoi, itos = build_vocab([tokens])

    print("\nVocabulary (stoi):")
    print(stoi)

    ids = encode(tokens, stoi)
    print("\nEncoded IDs:")
    print(ids)

    decoded_tokens = decode(ids, itos)
    print("\nDecoded tokens:")
    print(decoded_tokens)
