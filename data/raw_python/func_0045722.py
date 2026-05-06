def _upcase_word(text, pos):
    """Uppercase the current (or following) word."""
    text, new_pos = _forward_word(text, pos)
    return text[:pos] + text[pos:new_pos].upper() + text[new_pos:], new_pos