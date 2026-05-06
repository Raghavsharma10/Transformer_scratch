def merge(dicts):
    """
    Merges a list of dicts, summing their values.
    (Parallelized wrapper around `_count`)
    """
    chunks = [args for args in np.array_split(dicts, 20)]
    results = parallel(_count, chunks, n_jobs=-1)
    return _count(results)