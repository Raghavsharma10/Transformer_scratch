def thresholdForIdentity(identity, colors):
    """
    Get the best identity threshold for a specific identity value.

    @param identity: A C{float} nucleotide identity.
    @param colors: A C{list} of (threshold, color) tuples, where threshold is a
        C{float} and color is a C{str} to be used as a cell background. This
        is as returned by C{parseColors}.
    @return: The first C{float} threshold that the given identity is at least
        as big as.
    """
    for threshold, _ in colors:
        if identity >= threshold:
            return threshold
    raise ValueError('This should never happen! Last threshold is not 0.0?')