def train_tf(tokens_stream, out=None, **kwargs):
    """
    Train a map of term frequencies on a list of files (parallelized).
    """
    print('Counting terms...')
    results = parallel(count_tf, tokens_stream, n_jobs=-1)

    print('Merging...')
    tf = merge(results)

    if out is not None:
        with open(out, 'w') as f:
            json.dump(tf, f)

    return tf