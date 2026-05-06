def create_selfsim(oracle, method='rsfx'):
    """ Create self similarity matrix from attributes of a vmo object

    :param oracle: a encoded vmo object
    :param method:
        "comp":use the compression codes
        "sfx" - use suffix links
        "rsfx" - use reverse suffix links
        "lrs" - use LRS values
        "seg" - use patterns found
    :return: the created self-similarity matrix
    """

    len_oracle = oracle.n_states - 1
    mat = np.zeros((len_oracle, len_oracle))
    if method == 'com':
        if not oracle.code:
            print("Codes not generated. Generating codes with encode().")
            oracle.encode()
        ind = 0  # index
        for l, p in oracle.code:  # l for length, p for position

            if l == 0:
                inc = 1
            else:
                inc = l
            mat[range(ind, ind + inc), range(p - 1, p - 1 + inc)] = 1
            mat[range(p - 1, p - 1 + inc), range(ind, ind + inc)] = 1
            ind = ind + l
    elif method == 'sfx':
        for i, s in enumerate(oracle.sfx[1:]):
            if s != 0:
                mat[i][s - 1] = 1
                mat[s - 1][i] = 1
    elif method == 'rsfx':
        for cluster in oracle.latent:
            p = itertools.product(cluster, repeat=2)
            for _p in p:
                mat[_p[0] - 1][_p[1] - 1] = 1
    elif method == 'lrs':
        for i, l in enumerate(oracle.lrs[1:]):
            if l != 0:
                s = oracle.sfx[i + 1]
                mat[range((s - l) + 1, s + 1), range(i - l + 1, i + 1)] = 1
                mat[range(i - l + 1, i + 1), range((s - l) + 1, s + 1)] = 1
    elif method == 'seg':
        seg = oracle.segment
        ind = 0
        for l, p in seg:  # l for length, p for position

            if l == 0:
                inc = 1
            else:
                inc = l
            mat[range(ind, ind + inc), range(p - 1, p - 1 + inc)] = 1
            mat[range(p - 1, p - 1 + inc), range(ind, ind + inc)] = 1
            ind = ind + l

    return mat