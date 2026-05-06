def decode_vlqs(s):
    """
    Decode str `s` into a list of integers.
    """

    ints = []
    i = 0
    shift = 0

    for c in s:
        raw = B64_INT[c]
        cont = VLQ_CONT & raw
        i = ((VLQ_BASE_MASK & raw) << shift) | i
        shift += VLQ_SHIFT
        if not cont:
            sign = -1 if 1 & i else 1
            ints.append((i >> 1) * sign)
            i = 0
            shift = 0

    return tuple(ints)