def _downcase_word(text, pos):
    """Lowercase the current (or following) word."""
    text, new_pos = _forward_word(text, pos)
    return text[:pos] + text[pos:new_pos].lower() + text[new_pos:], new_pos