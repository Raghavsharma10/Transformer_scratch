def _kill_word(text, pos):
    """
    Kill from pos to the end of the current word, or if between words,
    to the end of the next word. Word boundaries are the same as those
    used by _forward_word.
    """
    text, end_pos = _forward_word(text, pos)
    return text[:pos] + text[end_pos:], pos