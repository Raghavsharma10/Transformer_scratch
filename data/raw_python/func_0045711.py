def _backward_delete_char(text, pos):
    """Delete the character behind pos."""
    if pos == 0:
        return text, pos
    return text[:pos - 1] + text[pos:], pos - 1