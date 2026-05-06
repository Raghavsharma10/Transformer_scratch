def tree(string, token=[WORD, POS, CHUNK, PNP, REL, ANCHOR, LEMMA]):
    """ Transforms the output of parse() into a Text object.
        The token parameter lists the order of tags in each token in the input string.
    """
    return Text(string, token)