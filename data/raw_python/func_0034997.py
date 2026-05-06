def predict(oracle, context, ab=None, verbose=False):
    """Single symbolic prediction given a context, an oracle and an alphabet.

    :param oracle: a learned vmo object from a symbolic sequence.
    :param context: the context precedes the predicted symbol
    :param ab: alphabet
    :param verbose: to show if the context if pruned or not
    :return: a probability distribution over the alphabet for the prediction.
    """
    if verbose:
        print("original context: ", context)
    if ab is None:
        ab = oracle.get_alphabet()

    _b, _s, context = _test_context(oracle, context)
    _lrs = [oracle.lrs[k] for k in oracle.rsfx[_s]]
    context_state = []
    while not context_state:
        for _i, _l in enumerate(_lrs):
            if _l >= len(context):
                context_state.append(oracle.rsfx[_s][_i])
        if context_state:
            break
        else:
            context = context[1:]
            _b, _s = oracle.accept(context)
            _lrs = [oracle.lrs[k] for k in oracle.rsfx[_s]]
    if verbose:
        print("final context: ", context)
        print("context_state: ", context_state)
    d_count = len(ab)
    hist = [1.0] * len(ab)  # initialize all histograms with 1s.

    trn_data = [oracle.data[n] for n in oracle.trn[_s]]
    for k in trn_data:
        hist[ab[k]] += 1.0
        d_count += 1.0

    for i in context_state:
        d_count, hist = _rsfx_count(oracle, i, d_count, hist, ab)

    return [hist[idx] / d_count for idx in range(len(hist))], context