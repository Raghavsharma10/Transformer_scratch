def _getStrippedValue(value, strip):
    """Like the strip() string method, except the strip argument describes
    different behavior:

    If strip is None, whitespace is stripped.

    If strip is a string, the characters in the string are stripped.

    If strip is False, nothing is stripped."""
    if strip is None:
        value = value.strip() # Call strip() with no arguments to strip whitespace.
    elif isinstance(strip, str):
        value = value.strip(strip) # Call strip(), passing the strip argument.
    elif strip is False:
        pass # Don't strip anything.
    return value