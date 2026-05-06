def wrap(text, width=70, **kwargs):
    """Wrap multiple paragraphs of text, returning a list of wrapped lines.

    Reformat the multiple paragraphs  'text' so they fit in lines of no
    more than 'width' columns, and return a list of wrapped lines.  By
    default, tabs in 'text' are expanded with string.expandtabs(), and
    all other whitespace characters (including newline) are converted to
    space.  See ParagraphWrapper class for available keyword args to customize
    wrapping behaviour.
    """
    w = ParagraphWrapper(width=width, **kwargs)
    return w.wrap(text)