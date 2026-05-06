def _hardClip(sequence, quality, cigartuples):
    """
    Hard clip (if necessary) a sequence.

    @param sequence: A C{str} nucleotide sequence.
    @param quality: A C{str} quality string, or a C{list} of C{int} quality
        values as returned by pysam, or C{None} if the SAM file had a '*'
        for the quality string (which pysam converts to C{None}).
    @param cigartuples: An iterable of (operation, length) tuples, detailing
        the alignment, as per the SAM specification.
    @return: A 3-tuple consisting of
            1) a hard-clipped C{str} sequence if hard-clipping is indicated by
               the CIGAR operations.
            2) a hard-clipped quality C{str} or C{list} (depending on what
               type we were passed) if hard-clipping is indicated by the CIGAR
               operations.
            3) a Boolean, C{True} if hard clipping was performed by this
               function or C{False} if the hard clipping had already been
               done.
    """
    hardClipCount = cigarLength = 0
    for (operation, length) in cigartuples:
        hardClipCount += operation == CHARD_CLIP
        cigarLength += length if operation in _CONSUMES_QUERY else 0

    sequenceLength = len(sequence)
    if quality is not None:
        assert sequenceLength == len(quality)
    clipLeft = clipRight = 0
    clippedSequence = sequence
    clippedQuality = quality

    if sequenceLength > cigarLength:
        alreadyClipped = False
    else:
        assert sequenceLength == cigarLength
        alreadyClipped = True

    if hardClipCount == 0:
        pass
    elif hardClipCount == 1:
        # Hard clip either at the start or the end.
        if cigartuples[0][0] == CHARD_CLIP:
            if not alreadyClipped:
                clipLeft = cigartuples[0][1]
                clippedSequence = sequence[clipLeft:]
                if quality is not None:
                    clippedQuality = quality[clipLeft:]
        elif cigartuples[-1][0] == CHARD_CLIP:
            if not alreadyClipped:
                clipRight = cigartuples[-1][1]
                clippedSequence = sequence[:-clipRight]
                if quality is not None:
                    clippedQuality = quality[:-clipRight]
        else:
            raise ValueError(
                'Invalid CIGAR tuples (%s) contains hard-clipping operation '
                'that is neither at the start nor the end of the sequence.' %
                (cigartuples,))
    elif hardClipCount == 2:
        # Hard clip at both the start and end.
        assert cigartuples[0][0] == cigartuples[-1][0] == CHARD_CLIP
        if not alreadyClipped:
            clipLeft, clipRight = cigartuples[0][1], cigartuples[-1][1]
            clippedSequence = sequence[clipLeft:-clipRight]
            if quality is not None:
                clippedQuality = quality[clipLeft:-clipRight]
    else:
        raise ValueError(
            'Invalid CIGAR tuples (%s) specifies hard-clipping %d times (2 '
            'is the maximum).' % (cigartuples, hardClipCount))

    weClipped = bool(clipLeft or clipRight)

    if weClipped:
        assert not alreadyClipped
        if len(clippedSequence) + clipLeft + clipRight != sequenceLength:
            raise ValueError(
                'Sequence %r (length %d) clipped to %r (length %d), but the '
                'difference between these two lengths (%d) is not equal to '
                'the sum (%d) of the left and right clip lengths (%d and %d '
                'respectively). CIGAR tuples: %s' %
                (sequence, len(sequence),
                 clippedSequence, len(clippedSequence),
                 abs(len(sequence) - len(clippedSequence)),
                 clipLeft + clipRight, clipLeft, clipRight, cigartuples))
    else:
        assert len(clippedSequence) == sequenceLength
        if quality is not None:
            assert len(clippedQuality) == sequenceLength

    return clippedSequence, clippedQuality, weClipped