def predict_fixation_duration(
        durations, angles, length_diffs, dataset=None, params=None):
    """
    Fits a non-linear piecewise regression to fixtaion durations for a fixmat.

    Returns corrected fixation durations.
    """
    if dataset is None:
        dataset = np.ones(durations.shape)
    corrected_durations = np.nan * np.ones(durations.shape)
    for i, ds in enumerate(np.unique(dataset)):
        e = lambda v, x, y, z: (leastsq_dual_model(x, z, *v) - y)
        v0 = [120, 220.0, -.1, 0.5, .1, .1]
        id_ds = dataset == ds
        idnan = (
            ~np.isnan(angles)) & (
            ~np.isnan(durations)) & (
            ~np.isnan(length_diffs))
        v, s = leastsq(
            e, v0, args=(
                angles[
                    idnan & id_ds], durations[
                    idnan & id_ds], length_diffs[
                    idnan & id_ds]), maxfev=10000)
        corrected_durations[id_ds] = (durations[id_ds] -
                                      (leastsq_dual_model(angles[id_ds], length_diffs[id_ds], *v)))
        if params is not None:
            params['v' + str(i)] = v
            params['s' + str(i)] = s
    return corrected_durations