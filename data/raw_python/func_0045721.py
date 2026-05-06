def _unix_word_rubout(text, pos):
    """
    Kill the word behind pos, using white space as a word boundary.
    """
    words = text[:pos].rsplit(None, 1)
    if len(words) < 2:
        return text[pos:], 0
    else:
        index = text.rfind(words[1], 0, pos)
        return text[:index] + text[pos:], index