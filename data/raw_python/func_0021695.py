def tokenize(string):
    """Match and yield all the tokens of the input string."""
    for match in TOKENS_REGEX.finditer(string):
        yield Token(match.lastgroup, match.group().strip(), match.span())