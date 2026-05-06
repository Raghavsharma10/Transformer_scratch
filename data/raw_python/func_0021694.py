def parse_nbt(literal):
    """Parse a literal nbt string and return the resulting tag."""
    parser = Parser(tokenize(literal))
    tag = parser.parse()

    cursor = parser.token_span[1]
    leftover = literal[cursor:]
    if leftover.strip():
        parser.token_span = cursor, cursor + len(leftover)
        raise parser.error(f'Expected end of string but got {leftover!r}')

    return tag