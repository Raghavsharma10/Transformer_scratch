def _capitalize_word(text, pos):
    """Capitalize the current (or following) word."""
    while pos < len(text) and not text[pos].isalnum():
        pos += 1
    if pos < len(text):
        text = text[:pos] + text[pos].upper() + text[pos + 1:]
    while pos < len(text) and text[pos].isalnum():
        pos += 1
    return text, pos