def pairwise_distances(dist_list, earth_mover_dist=True, energy_dist=True):
    """Applies statistical_distances to each unique pair of distribution
    samples in dist_list.

    Parameters
    ----------
    dist_list: list of 1d arrays
    earth_mover_dist: bool, optional
        Passed to statistical_distances.
    energy_dist: bool, optional
        Passed to statistical_distances.

    Returns
    -------
    ser: pandas Series object
        Values are statistical distances. Index levels are:
        calculation type: name of statistical distance.
        run: tuple containing the index in dist_list of the pair of samples
        arrays from which the statistical distance was computed.
    """
    out = []
    index = []
    for i, samp_i in enumerate(dist_list):
        for j, samp_j in enumerate(dist_list):
            if j < i:
                index.append(str((i, j)))
                out.append(statistical_distances(
                    samp_i, samp_j, earth_mover_dist=earth_mover_dist,
                    energy_dist=energy_dist))
    columns = ['ks pvalue', 'ks distance']
    if earth_mover_dist:
        columns.append('earth mover distance')
    if energy_dist:
        columns.append('energy distance')
    ser = pd.DataFrame(out, index=index, columns=columns).unstack()
    ser.index.names = ['calculation type', 'run']
    return ser