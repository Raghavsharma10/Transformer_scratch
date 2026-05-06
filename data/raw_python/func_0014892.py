def rexGroups(rex):
    """Return the named groups in a regular expression (compiled or as string)
    in occuring order.

    >>> rexGroups(r'(?P<name>\w+) +(?P<surname>\w+)')
    ('name', 'surname')

    """
    if isinstance(rex,basestring): rex = re.compile(rex)
    return zip(*sorted([(n,g) for (g,n) in rex.groupindex.items()]))[1]