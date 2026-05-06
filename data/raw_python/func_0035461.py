def improvise(oracle, seq_len, k=1, LRS=0, weight=None, continuity=1):
    """ Given an oracle and length, generate an improvised sequence of the given length.

    :param oracle: an indexed vmo object
    :param seq_len: the length of the returned improvisation sequence
    :param k: the starting improvisation time step in oracle
    :param LRS: the length of minimum longest repeated suffixes allowed to jump
    :param weight: if None, jump to possible candidate time step uniformly, if "lrs", the probability is proportional
    to the LRS of each candidate time step
    :param continuity: the number of time steps guaranteed to continue before next jump is executed
    :return: the improvised sequence
    """

    s = []
    if k + continuity < oracle.n_states - 1:
        s.extend(range(k, k + continuity))
        k = s[-1]
        seq_len -= continuity

    while seq_len > 0:
        s.append(improvise_step(oracle, k, LRS, weight))
        k = s[-1]
        if k + 1 < oracle.n_states - 1:
            k += 1
        else:
            k = 1
        if k + continuity < oracle.n_states - 1:
            s.extend(range(k, k + continuity))
            seq_len -= continuity
        k = s[-1]
        seq_len -= 1

    return s