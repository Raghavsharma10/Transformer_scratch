def fill(text, width=70, **kwargs):
    """Fill multiple paragraphs of text, returning a new string.

    Reformat multiple paragraphs in 'text' to fit in lines of no more
    than 'width' columns, and return a new string containing the entire
    wrapped text.  As with wrap(), tabs are expanded and other
    whitespace characters converted to space.  See ParagraphWrapper class for
    available keyword args to customize wrapping behaviour.
    """
    w = ParagraphWrapper(width=width, **kwargs)
    return w.fill(text)