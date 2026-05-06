def findPrimerBidiLimits(primer, seq):
    """
    Report the extreme (inner) offsets of primer in a sequence and its
    reverse complement.

    @param primer: A C{str} primer sequence.
    @param seq: A BioPython C{Bio.Seq} sequence.

    @return: A C{tuple} of two C{int} offsets. The first is a (zero-based)
        offset into the sequence that is beyond the first instance (if any)
        of the primer. The second is an offset into the original sequence of
        the beginning of the last instance of the primer in the reverse
        complemented sequence.

        In other words, if you wanted to chop all instances of a primer
        out of a sequence from the start and the end (when reverse
        complemented) you'd call this function and do something like this:

          start, end = findPrimerBidiLimits(primer, seq)
          seq = seq[start:end]
    """
    forward, reverse = findPrimerBidi(primer, seq)
    if forward:
        start = forward[-1] + len(primer)
        end = len(seq)
        for offset in reverse:
            if offset >= start:
                end = offset
                break
    else:
        start = 0
        end = reverse[0] if reverse else len(seq)

    return start, end