def ior_effect(durations, angle_diffs, length_diffs,
               summary_stat=np.mean, parallel=True, min_samples=20):
    """
    Computes a measure of fixation durations at delta angle and delta
    length combinations.
    """
    raster = np.empty((len(e_dist) - 1, len(e_angle) - 1), dtype=object)
    for a, (a_low, a_upp) in enumerate(zip(e_angle[:-1], e_angle[1:])):
        for d, (d_low, d_upp) in enumerate(zip(e_dist[:-1], e_dist[1:])):
            idx = ((d_low <= length_diffs) & (length_diffs < d_upp) &
                   (a_low <= angle_diffs) & (angle_diffs < a_upp))
            if sum(idx) < min_samples:
                raster[d, a] = np.array([np.nan])
            else:
                raster[d, a] = durations[idx]
    if parallel:
        p = pool.Pool(3)
        result = p.map(summary_stat, list(raster.flatten()))
        p.terminate()
    else:
        result = list(map(summary_stat, list(raster.flatten())))
    for idx, value in enumerate(result):
        i, j = np.unravel_index(idx, raster.shape)
        raster[i, j] = value
    return raster