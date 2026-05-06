def truncate_ellipsis(line, length=30):
    """Truncate a line to the specified length followed by ``...`` unless its shorter than length already."""

    l = len(line)
    return line if l < length else line[:length - 3] + "..."