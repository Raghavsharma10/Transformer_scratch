def encode_vlq(i):
    """
    Encode integer `i` into a VLQ encoded string.
    """

    # shift in the sign to least significant bit
    raw = (-i << 1) + 1 if i < 0 else i << 1
    if raw < VLQ_MULTI_CHAR:
        # short-circuit simple case as it doesn't need continuation
        return INT_B64[raw]

    result = []
    while raw:
        # assume continue
        result.append(raw & VLQ_BASE_MASK | VLQ_CONT)
        # shift out processed bits
        raw = raw >> VLQ_SHIFT
    # discontinue the last unit
    result[-1] &= VLQ_BASE_MASK
    return ''.join(INT_B64[i] for i in result)