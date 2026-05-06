def _backward_word(text, pos):
    """
    Move pos back to the start of the current or previous word. Words
    are composed of alphanumeric characters (letters and digits).
    """
    while pos > 0 and not text[pos - 1].isalnum():
        pos -= 1
    while pos > 0 and text[pos - 1].isalnum():
        pos -= 1
    return text, pos