def infix(tokens, operator_table):
    """Match an infix of an operator."""
    operator, matched_tokens = operator_table.infix.match(tokens)
    if operator:
        return TokenMatch(operator, None, matched_tokens)