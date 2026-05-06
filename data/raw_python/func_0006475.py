def salsa20_8(B):
    '''Salsa 20/8 stream cypher; Used by BlockMix. See http://en.wikipedia.org/wiki/Salsa20'''

    # Create a working copy
    x = B[:]

    # Expanded form of this code. The expansion is significantly faster but
    # this is much easier to understand
    # ROUNDS = (
    #     (4, 0, 12, 7),   (8, 4, 0, 9),    (12, 8, 4, 13),   (0, 12, 8, 18),
    #     (9, 5, 1, 7),    (13, 9, 5, 9),   (1, 13, 9, 13),   (5, 1, 13, 18),
    #     (14, 10, 6, 7),  (2, 14, 10, 9),  (6, 2, 14, 13),   (10, 6, 2, 18),
    #     (3, 15, 11, 7),  (7, 3, 15, 9),   (11, 7, 3, 13),   (15, 11, 7, 18),
    #     (1, 0, 3, 7),    (2, 1, 0, 9),    (3, 2, 1, 13),    (0, 3, 2, 18),
    #     (6, 5, 4, 7),    (7, 6, 5, 9),    (4, 7, 6, 13),    (5, 4, 7, 18),
    #     (11, 10, 9, 7),  (8, 11, 10, 9),  (9, 8, 11, 13),   (10, 9, 8, 18),
    #     (12, 15, 14, 7), (13, 12, 15, 9), (14, 13, 12, 13), (15, 14, 13, 18),
    # )
    #
    # for (destination, a1, a2, b) in ROUNDS:
    #     a = (x[a1] + x[a2]) & 0xffffffff
    #     x[destination] ^= ((a << b)  | (a >> (32 - b))) & 0xffffffff
    for i in (8, 6, 4, 2):
        a = (x[0] + x[12]) & 0xffffffff
        x[4] ^= ((a << 7) | (a >> 25))
        a = (x[4] + x[0]) & 0xffffffff
        x[8] ^= ((a << 9) | (a >> 23))
        a = (x[8] + x[4]) & 0xffffffff
        x[12] ^= ((a << 13) | (a >> 19))
        a = (x[12] + x[8]) & 0xffffffff
        x[0] ^= ((a << 18) | (a >> 14))
        a = (x[5] + x[1]) & 0xffffffff
        x[9] ^= ((a << 7) | (a >> 25))
        a = (x[9] + x[5]) & 0xffffffff
        x[13] ^= ((a << 9) | (a >> 23))
        a = (x[13] + x[9]) & 0xffffffff
        x[1] ^= ((a << 13) | (a >> 19))
        a = (x[1] + x[13]) & 0xffffffff
        x[5] ^= ((a << 18) | (a >> 14))
        a = (x[10] + x[6]) & 0xffffffff
        x[14] ^= ((a << 7) | (a >> 25))
        a = (x[14] + x[10]) & 0xffffffff
        x[2] ^= ((a << 9) | (a >> 23))
        a = (x[2] + x[14]) & 0xffffffff
        x[6] ^= ((a << 13) | (a >> 19))
        a = (x[6] + x[2]) & 0xffffffff
        x[10] ^= ((a << 18) | (a >> 14))
        a = (x[15] + x[11]) & 0xffffffff
        x[3] ^= ((a << 7) | (a >> 25))
        a = (x[3] + x[15]) & 0xffffffff
        x[7] ^= ((a << 9) | (a >> 23))
        a = (x[7] + x[3]) & 0xffffffff
        x[11] ^= ((a << 13) | (a >> 19))
        a = (x[11] + x[7]) & 0xffffffff
        x[15] ^= ((a << 18) | (a >> 14))
        a = (x[0] + x[3]) & 0xffffffff
        x[1] ^= ((a << 7) | (a >> 25))
        a = (x[1] + x[0]) & 0xffffffff
        x[2] ^= ((a << 9) | (a >> 23))
        a = (x[2] + x[1]) & 0xffffffff
        x[3] ^= ((a << 13) | (a >> 19))
        a = (x[3] + x[2]) & 0xffffffff
        x[0] ^= ((a << 18) | (a >> 14))
        a = (x[5] + x[4]) & 0xffffffff
        x[6] ^= ((a << 7) | (a >> 25))
        a = (x[6] + x[5]) & 0xffffffff
        x[7] ^= ((a << 9) | (a >> 23))
        a = (x[7] + x[6]) & 0xffffffff
        x[4] ^= ((a << 13) | (a >> 19))
        a = (x[4] + x[7]) & 0xffffffff
        x[5] ^= ((a << 18) | (a >> 14))
        a = (x[10] + x[9]) & 0xffffffff
        x[11] ^= ((a << 7) | (a >> 25))
        a = (x[11] + x[10]) & 0xffffffff
        x[8] ^= ((a << 9) | (a >> 23))
        a = (x[8] + x[11]) & 0xffffffff
        x[9] ^= ((a << 13) | (a >> 19))
        a = (x[9] + x[8]) & 0xffffffff
        x[10] ^= ((a << 18) | (a >> 14))
        a = (x[15] + x[14]) & 0xffffffff
        x[12] ^= ((a << 7) | (a >> 25))
        a = (x[12] + x[15]) & 0xffffffff
        x[13] ^= ((a << 9) | (a >> 23))
        a = (x[13] + x[12]) & 0xffffffff
        x[14] ^= ((a << 13) | (a >> 19))
        a = (x[14] + x[13]) & 0xffffffff
        x[15] ^= ((a << 18) | (a >> 14))


    # Add the original values
    for i in xrange(0, 16):
        B[i] = (B[i] + x[i]) & 0xffffffff