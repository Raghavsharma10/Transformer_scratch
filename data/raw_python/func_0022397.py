def compare_digests(digest_1, digest_2, is_hex_1=True, is_hex_2=True, threshold=None):
    """
    computes bit difference between two nilsisa digests
    takes params for format, default is hex string but can accept list
    of 32 length ints
    Optimized method originally from https://gist.github.com/michelp/6255490

    If `threshold` is set, and the comparison will be less than
    `threshold`, then bail out early and return a value just below the
    threshold.  This is a speed optimization that accelerates
    comparisons of very different items; e.g. tests show a ~20-30% speed
    up.  `threshold` must be an integer in the range [-128, 128].

    """
    # if we have both hexes use optimized method
    if threshold is not None:
        threshold -= 128
        threshold *= -1
    if is_hex_1 and is_hex_2:
        bits =  0
        for i in range_(0, 63, 2):
            bits += POPC[255 & int(digest_1[i:i+2], 16) ^ int(digest_2[i:i+2], 16)]
            if threshold is not None and bits > threshold: break
        return 128 - bits
    else:
        # at least one of the inputs is a list of unsigned ints
        if is_hex_1:  digest_1 = convert_hex_to_ints(digest_1)
        if is_hex_2:  digest_2 = convert_hex_to_ints(digest_2)
        bit_diff = 0
        for i in range(len(digest_1)):
            bit_diff += POPC[255 & digest_1[i] ^ digest_2[i]]
            if threshold is not None and bit_diff > threshold: break
        return 128 - bit_diff