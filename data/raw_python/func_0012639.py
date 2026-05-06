def saccadic_momentum_effect(durations, forward_angle,
                             summary_stat=nanmean):
    """
    Computes the mean fixation duration at forward angles.
    """
    durations_per_da = np.nan * np.ones((len(e_angle) - 1,))
    for i, (bo, b1) in enumerate(zip(e_angle[:-1], e_angle[1:])):
        idx = (
            bo <= forward_angle) & (
            forward_angle < b1) & (
            ~np.isnan(durations))
        durations_per_da[i] = summary_stat(durations[idx])
    return durations_per_da