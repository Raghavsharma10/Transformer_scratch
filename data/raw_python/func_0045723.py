def edit(text, pos, key):
    """
    Process a key input in the context of a line, and return the
    resulting text and cursor position.

    `text' and `key' must be of type str or unicode, and `pos' must be
    an int in the range [0, len(text)].

    If `key' is in keys(), the corresponding command is executed on the
    line. Otherwise, if `key' is a single character, that character is
    inserted at the cursor position. If neither condition is met, `text'
    and `pos' are returned unmodified.
    """
    if key in _key_bindings:
        return _key_bindings[key](text, pos)
    elif len(key) == 1:
        return text[:pos] + key + text[pos:], pos + 1
    else:
        return text, pos