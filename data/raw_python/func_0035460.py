def improvise_step(oracle, i, lrs=0, weight=None, prune=False):
    """ Given the current time step, improvise (generate) the next time step based on the oracle structure.

    :param oracle: an indexed vmo object
    :param i: current improvisation time step
    :param lrs: the length of minimum longest repeated suffixes allowed to jump
    :param weight: if None, jump to possible candidate time step uniformly, if "lrs", the probability is proportional
    to the LRS of each candidate time step
    :param prune: whether to prune improvisation steps based on regular beat structure or not
    :return: the next time step
    """

    if prune:
        prune_list = range(i % prune, oracle.n_states - 1, prune)
        trn_link = [s + 1 for s in oracle.latent[oracle.data[i]] if
                    (oracle.lrs[s] >= lrs and
                     (s + 1) < oracle.n_states) and
                    s in prune_list]
    else:
        trn_link = [s + 1 for s in oracle.latent[oracle.data[i]] if
                    (oracle.lrs[s] >= lrs and (s + 1) < oracle.n_states)]
    if not trn_link:
        if i == oracle.n_states - 1:
            n = 1
        else:
            n = i + 1
    else:
        if weight == 'lrs':
            lrs_link = [oracle.lrs[s] for s in
                        oracle.latent[oracle.data[i]] if
                        (oracle.lrs[s] >= lrs and (s + 1) < oracle.n_states)]
            lrs_pop = list(itertools.chain.from_iterable(itertools.chain.from_iterable(
                [[[i] * _x for (i, _x) in zip(trn_link, lrs_link)]])))
            n = np.random.choice(lrs_pop)
        else:
            n = trn_link[int(np.floor(random.random() * len(trn_link)))]
    return n