def compare_hexdigests( digest1, digest2 ):
    """Compute difference in bits between digest1 and digest2
       returns -127 to 128; 128 is the same, -127 is different"""
    # convert to 32-tuple of unsighed two-byte INTs
    digest1 = tuple([int(digest1[i:i+2],16) for i in range(0,63,2)])
    digest2 = tuple([int(digest2[i:i+2],16) for i in range(0,63,2)])
    bits = 0
    for i in range(32):
        bits += POPC[255 & digest1[i] ^ digest2[i]]
    return 128 - bits