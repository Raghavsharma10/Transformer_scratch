def _query_k(k, i, P, oracle, query, trn, state_cache, dist_cache, smooth=False, D=None, weight=0.5):
    """A helper function for query-matching function`s iteration over observations.
    
    Args:
        k - index of the candidate path
        i - index of the frames of the observations
        P - the path matrix of size K x N, K the number for paths initiated, 
            N the frame number of observations
        oracle - an encoded oracle
        query - observations matrix (numpy array) of dimension N x D. 
                D the dimension of the observation.
        trn - function handle of forward links vector gathering
        state_cache - a list storing the states visited during the for loop for k
        dist_cache - a list of the same lenth as oracle storing the 
                    distance calculated between the current observation and states 
                    in the oracle
        smooth - whether to enforce a preference on continuation or not
        D - Self-similarity matrix, required if smooth is set to True
        weight - the weight between continuation or jumps (1.0 for certain continuation)
    
    """

    _trn = trn(oracle, P[i - 1][k])
    t = list(itertools.chain.from_iterable([oracle.latent[oracle.data[j]] for j in _trn]))
    _trn_unseen = [_t for _t in _trn if _t not in state_cache]
    state_cache.extend(_trn_unseen)

    if _trn_unseen:
        t_unseen = list(itertools.chain.from_iterable([oracle.latent[oracle.data[j]] for j in _trn_unseen]))
        dist_cache[t_unseen] = _dist_obs_oracle(oracle, query[i], t_unseen)
    dvec = dist_cache[t]
    if smooth and P[i - 1][k] < oracle.n_states - 1:
        dvec = dvec * (1.0 - weight) + weight * np.array([D[P[i - 1][k]][_t - 1] for _t in t])
    _m = np.argmin(dvec)
    return t[_m], dvec[_m]