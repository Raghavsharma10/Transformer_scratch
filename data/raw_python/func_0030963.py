def findPrimer(primer, seq):
    """
    Look for a primer sequence.

    @param primer: A C{str} primer sequence.
    @param seq: A BioPython C{Bio.Seq} sequence.

    @return: A C{list} of zero-based offsets into the sequence at which the
        primer can be found. If no instances are found, return an empty
        C{list}.
    """
    offsets = []
    seq = seq.upper()
    primer = primer.upper()
    primerLen = len(primer)
    discarded = 0
    offset = seq.find(primer)

    while offset > -1:
        offsets.append(discarded + offset)
        seq = seq[offset + primerLen:]
        discarded += offset + primerLen
        offset = seq.find(primer)

    return offsets