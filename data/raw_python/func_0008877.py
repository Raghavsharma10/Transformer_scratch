def regroup(catalog, eps, far=None, dist=norm_dist):
    """
    Regroup the islands of a catalog according to their normalised distance.
    Return a list of island groups. Sources have their (island,source) parameters relabeled.


    Parameters
    ----------
    catalog : str or object
        Either a filename to read into a source list, or a list of objects with the following properties[units]:
        ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]

    eps : float
        maximum normalised distance within which sources are considered to be grouped

    far : float
        (degrees) sources that are further than this distance appart will not be grouped, and will not be tested.
        Default = None.

    dist : func
        a function that calculates the distance between two sources must accept two SimpleSource objects.
        Default = :func:`AegeanTools.cluster.norm_dist`

    Returns
    -------
    islands : list
        A list of islands. Each island is a list of sources.

    See Also
    --------
    :func:`AegeanTools.cluster.norm_dist`
    """

    if isinstance(catalog, str):
        table = load_table(catalog)
        srccat = table_to_source_list(table)
    else:
        try:
            srccat = catalog
            _ = catalog[0].ra, catalog[0].dec, catalog[0].a, catalog[0].b, catalog[0].pa, catalog[0].peak_flux

        except AttributeError as e:
            log.error("catalog is not understood.")
            log.error("catalog: Should be a list of objects with the following properties[units]:\n" +
                      "ra[deg],dec[deg], a[arcsec],b[arcsec],pa[deg], peak_flux[any]")
            raise e

    log.info("Regrouping islands within catalog")
    log.debug("Calculating distances")

    if far is None:
        far = 0.5  # 10*max(a.a/3600 for a in srccat)

    srccat_array = np.rec.fromrecords(
        [(s.ra, s.dec, s.a, s.b, s.pa, s.peak_flux)
         for s in srccat],
        names=['ra', 'dec', 'a', 'b', 'pa', 'peak_flux'])
    groups = regroup_vectorized(srccat_array, eps=eps, far=far, dist=dist)
    groups = [[srccat[idx] for idx in group]
              for group in groups]

    islands = []
    # now that we have the groups, we relabel the sources to have (island,component) in flux order
    # note that the order of sources within an island list is not changed - just their labels
    for isle, group in enumerate(groups):
        for comp, src in enumerate(sorted(group, key=lambda x: -1*x.peak_flux)):
            src.island = isle
            src.source = comp
        islands.append(group)
    return islands