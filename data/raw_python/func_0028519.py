def bold(text: str) -> str:
    '''
    Wraps the given text with bold enable/disable ANSI sequences.
    '''
    return (style(text, bold=True, reset=False) +
            style('', bold=False, reset=False))