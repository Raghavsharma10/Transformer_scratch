def _dyad_update(y, c):  # pylint:disable=too-many-locals
    # This function has many locals so it can be compared
    # with the original algorithm.
    """
    Inner function of the fast distance covariance.

    This function is compiled because otherwise it would become
    a bottleneck.

    """
    n = y.shape[0]
    gamma = np.zeros(n, dtype=c.dtype)

    # Step 1: get the smallest l such that n <= 2^l
    l_max = int(math.ceil(np.log2(n)))

    # Step 2: assign s(l, k) = 0
    s_len = 2 ** (l_max + 1)
    s = np.zeros(s_len, dtype=c.dtype)

    pos_sums = np.arange(l_max)
    pos_sums[:] = 2 ** (l_max - pos_sums)
    pos_sums = np.cumsum(pos_sums)

    # Step 3: iteration
    for i in range(1, n):

        # Step 3.a: update s(l, k)
        for l in range(l_max):
            k = int(math.ceil(y[i - 1] / 2 ** l))
            pos = k - 1

            if l > 0:
                pos += pos_sums[l - 1]

            s[pos] += c[i - 1]

        # Steps 3.b and 3.c
        for l in range(l_max):
            k = int(math.floor((y[i] - 1) / 2 ** l))
            if k / 2 > math.floor(k / 2):
                pos = k - 1
                if l > 0:
                    pos += pos_sums[l - 1]

                gamma[i] = gamma[i] + s[pos]

    return gamma