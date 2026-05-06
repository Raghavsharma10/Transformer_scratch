def getvalue(x):
    """Return the single value of x or raise TypError if more than one value."""
    if isrepeating(x):
        raise TypeError(
            "Ambiguous call to getvalue for %r which has more than one value."
            % x)

    for value in getvalues(x):
        return value