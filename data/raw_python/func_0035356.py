def save_segments(outfile, boundaries, beat_intervals, labels=None):
    """Save detected segments to a .lab file.

    :parameters:
        - outfile : str
            Path to output file

        - boundaries : list of int
            Beat indices of detected segment boundaries

        - beat_intervals : np.ndarray [shape=(n, 2)]
            Intervals of beats

        - labels : None or list of str
            Labels of detected segments
    """

    if labels is None:
        labels = [('Seg#%03d' % idx) for idx in range(1, len(boundaries))]

    times = [beat_intervals[beat, 0] for beat in boundaries[:-1]]
    times.append(beat_intervals[-1, -1])

    with open(outfile, 'w') as f:
        for idx, (start, end, lab) in enumerate(zip(times[:-1],
                                                    times[1:],
                                                    labels), 1):
            f.write('%.3f\t%.3f\t%s\n' % (start, end, lab))