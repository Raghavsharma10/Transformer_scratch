def parse_argument(string: str) -> Union[str, Tuple[str, str]]:
    """Return a single value for a string understood as a positional
    argument or a |tuple| containing a keyword and its value for a
    string understood as a keyword argument.

    |parse_argument| is intended to be used as a helper function for
    function |execute_scriptfunction| only.  See the following
    examples to see which types of keyword arguments |execute_scriptfunction|
    covers:

    >>> from hydpy.exe.commandtools import parse_argument
    >>> parse_argument('x=3')
    ('x', '3')
    >>> parse_argument('"x=3"')
    '"x=3"'
    >>> parse_argument("'x=3'")
    "'x=3'"
    >>> parse_argument('x="3==3"')
    ('x', '"3==3"')
    >>> parse_argument("x='3==3'")
    ('x', "'3==3'")
    """
    idx_equal = string.find('=')
    if idx_equal == -1:
        return string
    idx_quote = idx_equal+1
    for quote in ('"', "'"):
        idx = string.find(quote)
        if -1 < idx < idx_quote:
            idx_quote = idx
    if idx_equal < idx_quote:
        return string[:idx_equal], string[idx_equal+1:]
    return string