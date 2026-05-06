def keyword(tokens, expected):
    """Case-insensitive keyword match."""
    try:
        token = next(iter(tokens))
    except StopIteration:
        return

    if token and token.name == "symbol" and token.value.lower() == expected:
        return TokenMatch(None, token.value, (token,))