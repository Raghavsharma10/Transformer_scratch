def addORFs(fig, seq, minX, maxX, offsetAdjuster):
    """
    fig is a matplotlib figure.
    seq is a Bio.Seq.Seq.
    minX: the smallest x coordinate.
    maxX: the largest x coordinate.
    featureEndpoints: an array of features as returned by addFeatures (may be
        empty).
    offsetAdjuster: a function to adjust feature X axis offsets for plotting.
    """
    for frame in range(3):
        target = seq[frame:]
        for (codons, codonType, color) in (
                (START_CODONS, 'start', 'green'),
                (STOP_CODONS, 'stop', 'red')):
            offsets = list(map(offsetAdjuster, findCodons(target, codons)))
            if offsets:
                fig.plot(offsets, np.tile(frame, len(offsets)), marker='.',
                         markersize=4, color=color, linestyle='None')

    fig.axis([minX, maxX, -1, 3])
    fig.set_yticks(np.arange(3))
    fig.set_ylabel('Frame', fontsize=17)
    fig.set_title('Subject start (%s) and stop (%s) codons' % (
        ', '.join(sorted(START_CODONS)), ', '.join(sorted(STOP_CODONS))),
        fontsize=20)