def prepare_data(fm, max_back, dur_cap=700):
    '''
    Computes angle and length differences up to given order and deletes
    suspiciously long fixations.

    Input
        fm: Fixmat
            Fixmat for which to comput angle and length differences
        max_back: Int
            Computes delta angle and amplitude up to order max_back.
        dur_cap: Int
            Longest allowed fixation duration

    Output
        fm: Fixmat
            Filtered fixmat that aligns to the other outputs.
        durations: ndarray
            Duration for each fixation in fm
        forward_angle:
            Angle between previous and next saccade.

    '''
    durations = np.roll(fm.end - fm.start, 1).astype(float)
    angles, lengths, ads, lds = anglendiff(fm, roll=max_back, return_abs=True)
    # durations and ads are aligned in a way that an entry in ads
    # encodes the angle of the saccade away from a fixation in
    # durations
    forward_angle = abs(reshift(ads[0])).astype(float)
    ads = [abs(reshift(a)) for a in ads]
    # Now filter out weird fixation durations
    id_in = durations > dur_cap
    durations[id_in] = np.nan
    forward_angle[id_in] = np.nan
    return fm, durations, forward_angle, ads, lds