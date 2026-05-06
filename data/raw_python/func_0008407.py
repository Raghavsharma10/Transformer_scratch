def match(pattern, sentence, *args, **kwargs):
    """ Returns the first match found in the given sentence, or None.
    """
    return compile(pattern, *args, **kwargs).match(sentence)