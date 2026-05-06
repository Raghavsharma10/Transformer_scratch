def _backward_kill_word(text, pos):
    """"
    Kill the word behind pos. Word boundaries are the same as those
    used by _backward_word.
    """
    text, new_pos = _backward_word(text, pos)
    return text[:new_pos] + text[pos:], new_pos