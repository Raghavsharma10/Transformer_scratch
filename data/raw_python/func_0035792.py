def tokens(istr):
    """
    Same as tokenize, but returns only tokens
    (and at all parantheses levels).
    """

    # make a list of all alphanumeric tokens
    toks = re.findall(r'[^\*\\\+\-\^\(\)]+\(?', istr)

    # remove the functions
    return [t for t in toks if not t.endswith('(')]