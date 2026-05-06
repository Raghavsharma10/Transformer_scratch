def blockmix_salsa8(BY, Yi, r):
    '''Blockmix; Used by SMix.'''

    start = (2 * r - 1) * 16
    X = BY[start:start + 16]                                      # BlockMix - 1

    for i in xrange(0, 2 * r):                                    # BlockMix - 2

        for xi in xrange(0, 16):                                  # BlockMix - 3(inner)
            X[xi] ^= BY[i * 16 + xi]

        salsa20_8(X)                                              # BlockMix - 3(outer)
        aod = Yi + i * 16                                         # BlockMix - 4
        BY[aod:aod + 16] = X[:16]

    for i in xrange(0, r):                                        # BlockMix - 6 (and below)
        aos = Yi + i * 32
        aod = i * 16
        BY[aod:aod + 16] = BY[aos:aos + 16]

    for i in xrange(0, r):
        aos = Yi + (i * 2 + 1) * 16
        aod = (i + r) * 16
        BY[aod:aod + 16] = BY[aos:aos + 16]