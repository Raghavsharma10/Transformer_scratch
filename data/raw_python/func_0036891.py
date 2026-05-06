def dumps(value,encoding=None):
    """dumps(object,encoding=None) -> string

    This function dumps a python object as a tnetstring.
    """
    #  This uses a deque to collect output fragments in reverse order,
    #  then joins them together at the end.  It's measurably faster
    #  than creating all the intermediate strings.
    #  If you're reading this to get a handle on the tnetstring format,
    #  consider the _gdumps() function instead; it's a standard top-down
    #  generator that's simpler to understand but much less efficient.
    q = deque()
    _rdumpq(q,0,value,encoding)
    return "".join(q)