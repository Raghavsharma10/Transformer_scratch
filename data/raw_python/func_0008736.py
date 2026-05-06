def classify_catalog(catalog):
    """
    Look at a list of sources and split them according to their class.

    Parameters
    ----------
    catalog : iterable
        A list or iterable object of {SimpleSource, IslandSource, OutputSource} objects, possibly mixed.
        Any other objects will be silently ignored.

    Returns
    -------
    components : list
        List of sources of type OutputSource

    islands : list
        List of sources of type IslandSource

    simples : list
        List of source of type SimpleSource
    """
    components = []
    islands = []
    simples = []
    for source in catalog:
        if isinstance(source, OutputSource):
            components.append(source)
        elif isinstance(source, IslandSource):
            islands.append(source)
        elif isinstance(source, SimpleSource):
            simples.append(source)
    return components, islands, simples