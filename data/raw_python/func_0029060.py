def distant_total_damped_rated_level(
            octave_frequencies,
            distance,
            temp,
            relhum,
            reference_distance=1.0):
    """
    Calculates the damped, A-rated total sound pressure level
    in a given distance, temperature and relative humidity
    from octave frequency sound pressure levels in a reference distance
    """
    damping_distance = distance - reference_distance
    sums = 0.0
    for band in OCTAVE_BANDS.keys():
        if band not in octave_frequencies:
            continue
        if octave_frequencies[band] is None:
            continue
        # distance-adjusted level per band
        distant_val = distant_level(
            reference_level=float(octave_frequencies[band]),
            distance=distance,
            reference_distance=reference_distance
        )
        # damping
        damp_per_meter = damping(
            temp=temp,
            relhum=relhum,
            freq=OCTAVE_BANDS[band][0])
        distant_val = distant_val - (damping_distance * damp_per_meter)
        # applyng A-rating
        distant_val += OCTAVE_BANDS[band][1]
        sums += pow(10.0, (distant_val / 10.0))
    level = 10.0 * math.log10(sums)
    return level