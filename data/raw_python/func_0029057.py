def total_rated_level(octave_frequencies):
    """
    Calculates the A-rated total sound pressure level
    based on octave band frequencies
    """
    sums = 0.0
    for band in OCTAVE_BANDS.keys():
        if band not in octave_frequencies:
            continue
        if octave_frequencies[band] is None:
            continue
        if octave_frequencies[band] == 0:
            continue
        sums += pow(10.0, ((float(octave_frequencies[band]) + OCTAVE_BANDS[band][1]) / 10.0))
    level = 10.0 * math.log10(sums)
    return level