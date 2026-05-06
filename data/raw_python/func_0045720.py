def _transpose_words(text, pos):
    """
    Drag the word before pos past the word after pos, moving pos over
    that word as well. If pos is at the end of text, this transposes the
    last two words in text.
    """
    text, end2 = _forward_word(text, pos)
    text, start2 = _backward_word(text, end2)
    text, start1 = _backward_word(text, start2)
    text, end1 = _forward_word(text, start1)
    if start1 == start2:
        return text, pos
    return text[:start1] + text[start2:end2] + text[end1:start2:] + \
        text[start1:end1] + text[end2:], end2