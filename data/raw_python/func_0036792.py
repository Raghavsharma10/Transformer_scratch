def train_idf(tokens_stream, out=None, **kwargs):
    """
    Train a IDF model on a list of files (parallelized).
    """
    idfs = parallel(count_idf, tokens_stream, n_jobs=-1)
    N = len(idfs) # n docs
    idf = merge(idfs)

    for k, v in idf.items():
        idf[k] = math.log(N/v)
        # v ~= N/(math.e ** idf[k])

    # Keep track of N to update IDFs
    idf['_n_docs'] = N

    if out is not None:
        with open(out, 'w') as f:
            json.dump(idf, f)

    return idf