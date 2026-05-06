def convert_raw_tuple(value_tuple, format_string):
    """
    Convert a tuple of raw values, according to the given line format.

    :param tuple value_tuple: the tuple of raw values
    :param str format_string: the format of the tuple
    :rtype: list of tuples
    """ 
    values = []
    for v, c in zip(value_tuple, format_string):
        if v is None:
            # append None
            values.append(v)
        elif c == u"s":
            # string
            values.append(v)
        elif c == u"S":
            # string, split using space as delimiter
            values.append([s for s in v.split(u" ") if len(s) > 0])
        elif c == u"i":
            # int
            values.append(int(v))
        elif c == u"U":
            # Unicode
            values.append(convert_unicode_field(v))
        elif c == u"A":
            # ASCII
            values.append(convert_ascii_field(v))
        #elif c == u"x":
        #    # ignore
        #    pass
    return tuple(values)