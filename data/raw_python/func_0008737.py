def island_itergen(catalog):
    """
    Iterate over a catalog of sources, and return an island worth of sources at a time.
    Yields a list of components, one island at a time

    Parameters
    ----------
    catalog : iterable
        A list or iterable of :class:`AegeanTools.models.OutputSource` objects.

    Yields
    ------
    group : list
        A list of all sources within an island, one island at a time.

    """
    # reverse sort so that we can pop the last elements and get an increasing island number
    catalog = sorted(catalog)
    catalog.reverse()
    group = []

    # using pop and keeping track of the list length ourselves is faster than
    # constantly asking for len(catalog)
    src = catalog.pop()
    c_len = len(catalog)
    isle_num = src.island
    while c_len >= 0:
        if src.island == isle_num:
            group.append(src)
            c_len -= 1
            if c_len <0:
                # we have just added the last item from the catalog
                # and there are no more to pop
                yield group
            else:
                src = catalog.pop()
        else:
            isle_num += 1
            # maybe there are no sources in this island so skip it
            if group == []:
                continue
            yield group
            group = []
    return