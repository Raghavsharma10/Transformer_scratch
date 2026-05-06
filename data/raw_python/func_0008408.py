def search(pattern, sentence, *args, **kwargs):
    """ Returns a list of all matches found in the given sentence.
    """
    return compile(pattern, *args, **kwargs).search(sentence)