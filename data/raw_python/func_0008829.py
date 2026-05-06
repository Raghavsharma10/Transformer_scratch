def intersect_regions(flist):
    """
    Construct a region which is the intersection of all regions described in the given
    list of file names.

    Parameters
    ----------
    flist : list
        A list of region filenames.

    Returns
    -------
    region : :class:`AegeanTools.regions.Region`
        The intersection of all regions, possibly empty.
    """
    if len(flist) < 2:
        raise Exception("Require at least two regions to perform intersection")
    a = Region.load(flist[0])
    for b in [Region.load(f) for f in flist[1:]]:
        a.intersect(b)
    return a