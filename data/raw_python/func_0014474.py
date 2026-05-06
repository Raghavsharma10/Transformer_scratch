def _roll_random(rng, n):
    "returns a random # from 0 to N-1"
    bits = util.bit_length(n-1)
    bytes = (bits + 7) // 8
    hbyte_mask = pow(2, bits % 8) - 1

    # so here's the plan:
    # we fetch as many random bits as we'd need to fit N-1, and if the
    # generated number is >= N, we try again.  in the worst case (N-1 is a
    # power of 2), we have slightly better than 50% odds of getting one that
    # fits, so i can't guarantee that this loop will ever finish, but the odds
    # of it looping forever should be infinitesimal.
    while True:
        x = rng.read(bytes)
        if hbyte_mask > 0:
            x = chr(ord(x[0]) & hbyte_mask) + x[1:]
        num = util.inflate_long(x, 1)
        if num < n:
            break
    return num