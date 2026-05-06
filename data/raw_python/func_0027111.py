def ellipsis(source, max_length):
    """Truncates a string to be at most max_length long."""
    if max_length == 0 or len(source) <= max_length:
        return source
    return source[: max(0, max_length - 3)] + "..."