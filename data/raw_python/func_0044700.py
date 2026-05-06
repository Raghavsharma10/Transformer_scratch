def multiline_repr(text, special_chars=('\n', '"')):
    """Get string representation for triple quoted context.

    Make string representation as normal except do not transform
    "special characters" into an escaped representation to support
    use of the representation in a triple quoted multi-line string
    context (to avoid escaping newlines and double quotes).

     Pass ``RAW_MULTILINE_CHARS`` as the ``special_chars`` when use
     context is a "raw" triple quoted string (to also avoid excaping
     backslashes).

    :param text: string
    :type text: str or unicode
    :param iterable special_chars: characters to remove/restore
    :returns: representation
    :rtype: str

    """
    try:
        char = special_chars[0]
    except IndexError:
        text = ascii(text)[2 if PY2 else 1:-1]
    else:
        text = char.join(
            multiline_repr(s, special_chars[1:]) for s in text.split(char))

    return text