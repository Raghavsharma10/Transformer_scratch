def smix(B, Bi, r, N, V, X):
    '''SMix; a specific case of ROMix. See scrypt.pdf in the links above.'''

    X[:32 * r] = B[Bi:Bi + 32 * r]                   # ROMix - 1

    for i in xrange(0, N):                           # ROMix - 2
        aod = i * 32 * r                             # ROMix - 3
        V[aod:aod + 32 * r] = X[:32 * r]
        blockmix_salsa8(X, 32 * r, r)                # ROMix - 4

    for i in xrange(0, N):                           # ROMix - 6
        j = X[(2 * r - 1) * 16] & (N - 1)            # ROMix - 7
        for xi in xrange(0, 32 * r):                 # ROMix - 8(inner)
            X[xi] ^= V[j * 32 * r + xi]

        blockmix_salsa8(X, 32 * r, r)                # ROMix - 9(outer)

    B[Bi:Bi + 32 * r] = X[:32 * r]