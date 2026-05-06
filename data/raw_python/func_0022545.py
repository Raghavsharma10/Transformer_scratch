def suffix(tokens, operator_table):
    """Match a suffix of an operator."""
    operator, matched_tokens = operator_table.suffix.match(tokens)
    if operator:
        return TokenMatch(operator, None, matched_tokens)