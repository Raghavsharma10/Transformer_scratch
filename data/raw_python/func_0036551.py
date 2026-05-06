def parallel(func, inputs, n_jobs, expand_args=False):
    """
    Convenience wrapper around joblib's parallelization.
    """
    if expand_args:
        return Parallel(n_jobs=n_jobs)(delayed(func)(*args) for args in inputs)
    else:
        return Parallel(n_jobs=n_jobs)(delayed(func)(arg) for arg in inputs)