def regroup_vectorized(srccat, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance.

    Assumes srccat is recarray-like for efficiency.
    Return a list of island groups.

    Parameters
    ----------
    srccat : np.rec.arry or pd.DataFrame
        Should have the following fields[units]:
        ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]
    eps : float
        maximum normalised distance within which sources are considered to be
        grouped
    far : float
        (degrees) sources that are further than this distance apart will not
        be grouped, and will not be tested.
        Default = 0.5.
    dist : func
        a function that calculates the distance between a source and each
        element of an array of sources.
        Default = :func:`AegeanTools.cluster.norm_dist`

    Returns
    -------
    islands : list of lists
        Each island contians integer indices for members from srccat
        (in descending dec order).
    """
    if far is None:
        far = 0.5  # 10*max(a.a/3600 for a in srccat)

    # most negative declination first
    # XXX: kind='mergesort' ensures stable sorting for determinism.
    #      Do we need this?
    order = np.argsort(srccat.dec, kind='mergesort')[::-1]
    # TODO: is it better to store groups as arrays even if appends are more
    #       costly?
    groups = [[order[0]]]
    for idx in order[1:]:
        rec = srccat[idx]
        # TODO: Find out if groups are big enough for this to give us a speed
        #       gain. If not, get distance to all entries in groups above
        #       decmin simultaneously.
        decmin = rec.dec - far
        for group in reversed(groups):
            # when an island's largest (last) declination is smaller than
            # decmin, we don't need to look at any more islands
            if srccat.dec[group[-1]] < decmin:
                # new group
                groups.append([idx])
            rafar = far / np.cos(np.radians(rec.dec))
            group_recs = np.take(srccat, group, mode='clip')
            group_recs = group_recs[abs(rec.ra - group_recs.ra) <= rafar]
            if len(group_recs) and dist(rec, group_recs).min() < eps:
                group.append(idx)
                break
        else:
            # new group
            groups.append([idx])

    # TODO?: a more numpy-like interface would return only an array providing
    #        the mapping:
    #    group_idx = np.empty(len(srccat), dtype=int)
    #    for i, group in enumerate(groups):
    #        group_idx[group] = i
    #    return group_idx
    return groups