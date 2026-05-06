def escape_string(text):
    """Remove problematic characters.

    Parameters
    ----------
    text
        A string with potentially problematic characters.

    Returns
    -------
    string
        The text with characters removed.


    Examples
    -------
    >>> s = r'hello_world$here'
    >>> escape_string(s) == r'helloworldhere'
    True
    """
    if text is None:
        return text


    TO_REMOVE = [r'&', r'%', r'$', r'#', r'_', r'{', r'}',
                 r'~', r'^', r'\\']

    for char in TO_REMOVE:
        text = text.replace(char, '')
    return text