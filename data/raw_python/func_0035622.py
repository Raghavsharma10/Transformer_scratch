def repr_compat(s):
    """
    Since Python 2 is annoying with unicode literals, and that we are
    enforcing the usage of unicode, this ensures the repr doesn't spew
    out the unicode literal prefix.
    """

    if unicode and isinstance(s, unicode):
        return repr(s)[1:]
    else:
        return repr(s)