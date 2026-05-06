def grade(adjective, suffix=COMPARATIVE):
    """ Returns the comparative or superlative form of the given (inflected) adjective.
    """
    b = predicative(adjective)
    # groß => großt, schön => schönst
    if suffix == SUPERLATIVE and b.endswith(("s", u"ß")):
        suffix = suffix[1:]
    # große => großere, schönes => schöneres
    return adjective[:len(b)] + suffix + adjective[len(b):]