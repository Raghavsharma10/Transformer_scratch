def statistical_distances(samples1, samples2, earth_mover_dist=True,
                          energy_dist=True):
    """Compute measures of the statistical distance between samples.

    Parameters
    ----------
    samples1: 1d array
    samples2: 1d array
    earth_mover_dist: bool, optional
        Whether or not to compute the Earth mover's distance between the
        samples.
    energy_dist: bool, optional
        Whether or not to compute the energy distance between the samples.

    Returns
    -------
    1d array
    """
    out = []
    temp = scipy.stats.ks_2samp(samples1, samples2)
    out.append(temp.pvalue)
    out.append(temp.statistic)
    if earth_mover_dist:
        out.append(scipy.stats.wasserstein_distance(samples1, samples2))
    if energy_dist:
        out.append(scipy.stats.energy_distance(samples1, samples2))
    return np.asarray(out)