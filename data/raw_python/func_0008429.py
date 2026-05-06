def keywords(s, top=10, **kwargs):
    """ Returns a sorted list of keywords in the given string.
    """
    return parser.find_keywords(s, top=top, frequency=parser.frequency)