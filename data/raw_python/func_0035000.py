def tracking(oracle, obs, trn_type=1, reverse_init=False, method='else', decay=1.0):
    """ Off-line tracking function using sub-optimal query-matching algorithm"""
    N = len(obs)
    if reverse_init:
        r_oracle = create_reverse_oracle(oracle)
        _ind = [r_oracle.n_states - rsfx for rsfx in r_oracle.rsfx[0][:]]
        init_ind = []
        for i in _ind:
            s = i
            while oracle.sfx[s] != 0:
                s = oracle.sfx[s]
            init_ind.append(s)
        K = r_oracle.num_clusters()
    else:
        init_ind = oracle.rsfx[0][:]
        K = oracle.num_clusters()

    P = np.zeros((N, K), dtype='int')
    T = np.zeros((N,), dtype='int')
    map_k_outer = partial(_query_k, oracle=oracle, query=obs)
    map_query = partial(_query_init, oracle=oracle, query=obs[0], method=method)
    #     map_query = partial(_query_init, oracle=oracle, query=obs[0], method)

    argmin = np.argmin

    P[0], C = zip(*map(map_query, init_ind))
    C = np.array(C)
    T[0] = P[0][argmin(C)]

    if trn_type == 1:
        trn = _create_trn_self
    elif trn_type == 2:
        trn = _create_trn_sfx_rsfx
    else:
        trn = _create_trn

    distance_cache = np.zeros(oracle.n_states)

    for i in range(1, N):  # iterate over the rest of query
        state_cache = []
        dist_cache = distance_cache

        map_k_inner = partial(map_k_outer, i=i, P=P, trn=trn, state_cache=state_cache, dist_cache=dist_cache)
        P[i], _c = zip(*map(map_k_inner, range(K)))
        C = decay * C + np.array(_c)
        T[i] = P[i][argmin(C)]

    return T