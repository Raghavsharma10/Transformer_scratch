def get_assemblies(names=[]):
    """read specified assemblies, or all if none specified, returning a
    dictionary of assembly-name: assembly.  See get_assembly()
    function for the structure of assembly data.

    >>> assemblies = get_assemblies(names=['GRCh37.p13'])
    >>> assy = assemblies['GRCh37.p13']

    >>> assemblies = get_assemblies()
    >>> 'GRCh38.p2' in assemblies
    True

    """

    if names == []:
        names = get_assembly_names()
    return {a['name']: a for a in (get_assembly(n) for n in names)}