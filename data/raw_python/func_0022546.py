def token_name(tokens, expected):
    """Match a token name (type)."""
    try:
        token = next(iter(tokens))
    except StopIteration:
        return

    if token and token.name == expected:
        return TokenMatch(None, token.value, (token,))