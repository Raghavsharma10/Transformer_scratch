def prefix(tokens, operator_table):
    """Match a prefix of an operator."""
    operator, matched_tokens = operator_table.prefix.match(tokens)
    if operator:
        return TokenMatch(operator, None, matched_tokens)