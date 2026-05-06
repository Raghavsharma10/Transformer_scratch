def hashes(peaks, f_width=F_WIDTH, t_gap=ROWS_PER_SECOND, t_width=2*ROWS_PER_SECOND):
    """
    Generator function for successive hashes calculated from a mono-channel
    time-domain audio signal as a set of tuples, (<long>, <int>). The <long>
    is an integral 64-bit hash so it can be used as a database ID, and
    the <int> is the frame number associated with the beginning of
    the time bin for the anchor point.

    The frequency window of each peak for constellation is +/- 1 octave

    Time gap and width recommendations:

    To calculate N seconds in rows (DTFT time windows):
        rows = N * (1 + (FREQ - FRAME_WIDTH) // FRAME_STRIDE)

    """
    for i, (t1, f1) in enumerate(peaks):

        # limit constellations to a window -- a box constrained by a min
        # and max time limit, and a min and max frequency bound
        t_min = t1 + t_gap
        t_max = t_min + t_width
        f_min = f1 - f_width // 2
        f_max = f1 + f_width // 2

        for t2, f2 in peaks[i:]:
            if t2 < t_min or f2 < f_min:
                continue
            elif t2 > t_max:
                break
            elif f2 < f_max:
                yield (_get_hash(f1, f2, t2 - t1), t1)