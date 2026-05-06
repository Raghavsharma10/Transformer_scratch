def getdict(source):
    """Returns a standard python Dict with computed values
    from the DynDict
    :param source: (DynDict) input
    :return: (dict) Containing computed values
    """
    std_dict = {}
    for var, val in source.iteritems():
        std_dict[var] = source[var]
    return std_dict