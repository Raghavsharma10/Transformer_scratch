def match_tokens(expected_tokens):
    """Generate a grammar function that will match 'expected_tokens' only."""
    if isinstance(expected_tokens, Token):
        # Match a single token.
        def _grammar_func(tokens):
            try:
                next_token = next(iter(tokens))
            except StopIteration:
                return

            if next_token == expected_tokens:
                return TokenMatch(None, next_token.value, (next_token,))

    elif isinstance(expected_tokens, tuple):
        # Match multiple tokens.
        match_len = len(expected_tokens)
        def _grammar_func(tokens):
            upcoming = tuple(itertools.islice(tokens, match_len))
            if upcoming == expected_tokens:
                return TokenMatch(None, None, upcoming)
    else:
        raise TypeError(
            "'expected_tokens' must be an instance of Token or a tuple "
            "thereof. Got %r." % expected_tokens)

    return _grammar_func