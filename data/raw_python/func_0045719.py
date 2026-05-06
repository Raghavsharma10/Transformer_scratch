def _transpose_chars(text, pos):
    """
    Drag the character before pos forward over the character at pos,
    moving pos forward as well. If pos is at the end of text, then this
    transposes the two characters before pos.
    """
    if len(text) < 2 or pos == 0:
        return text, pos
    if pos == len(text):
        return text[:pos - 2] + text[pos - 1] + text[pos - 2], pos
    return text[:pos - 1] + text[pos] + text[pos - 1] + text[pos + 1:], pos + 1