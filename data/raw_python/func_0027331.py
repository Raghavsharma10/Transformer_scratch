def pad(segment, size):
    """Add zeroes to a segment until it reaches a certain size.

    :param segment: the segment to pad
    :param size: the size to which to pad the segment
    """
    for i in range(size - len(segment)):
        segment.append(0)

    assert len(segment) == size