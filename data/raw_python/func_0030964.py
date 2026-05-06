def findPrimerBidi(primer, seq):
    """
    Look for a primer in a sequence and its reverse complement.

    @param primer: A C{str} primer sequence.
    @param seq: A BioPython C{Bio.Seq} sequence.

    @return: A C{tuple} of two lists. The first contains (zero-based)
        ascending offsets into the sequence at which the primer can be found.
        The second is a similar list ascending offsets into the original
        sequence where the primer matches the reverse complemented of the
        sequence. If no instances are found, the corresponding list in the
        returned tuple must be empty.
    """
    # Note that we reverse complement the primer to find the reverse
    # matches. This is much simpler than reverse complementing the sequence
    # because it allows us to use findPrimer and to deal with overlapping
    # matches correctly.
    forward = findPrimer(primer, seq)
    reverse = findPrimer(reverse_complement(primer), seq)
    return forward, reverse