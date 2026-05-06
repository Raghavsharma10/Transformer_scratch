def _forward_word(text, pos):
    """
    Move pos forward to the end of the next word. Words are composed of
    alphanumeric characters (letters and digits).
    """
    while pos < len(text) and not text[pos].isalnum():
        pos += 1
    while pos < len(text) and text[pos].isalnum():
        pos += 1
    return text, pos