def application(tokens):
    """Matches function call (application)."""
    tokens = iter(tokens)
    func = next(tokens)
    paren = next(tokens)

    if func and func.name == "symbol" and paren.name == "lparen":
        # We would be able to unambiguously parse function application with
        # whitespace between the function name and the lparen, but let's not
        # do that because it's unexpected in most languages.
        if func.end != paren.start:
            raise errors.EfilterParseError(
                start=func.start, end=paren.end,
                message="No whitespace allowed between function and paren.")

        return common.TokenMatch(None, func.value, (func, paren))