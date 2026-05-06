def _stringifyKeys(d):
    """
    Return a copy of C{d} with C{str} keys.

    @type d: C{dict} with C{unicode} keys.
    @rtype: C{dict} with C{str} keys.
    """
    return dict((k.encode('ascii'), v)  for (k, v) in d.iteritems())