def multi_keyword(tokens, keyword_parts):
    """Match a case-insensitive keyword consisting of multiple tokens."""
    tokens = iter(tokens)
    matched_tokens = []
    limit = len(keyword_parts)

    for idx in six.moves.range(limit):
        try:
            token = next(tokens)
        except StopIteration:
            return

        if (not token or token.name != "symbol" or
                token.value.lower() != keyword_parts[idx]):
            return

        matched_tokens.append(token)

    return TokenMatch(None, token.value, matched_tokens)