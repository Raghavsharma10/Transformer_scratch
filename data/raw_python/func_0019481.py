def data_gen(n_ops=100):
    """Yield data, while optionally burning compute cycles.

    Parameters
    ----------
    n_ops : int, default=100
        Number of operations to run between yielding data.

    Returns
    -------
    data : dict
        A object which looks like it might come from some
        machine learning problem, with X as features, and y as targets.
    """
    while True:
        X = np.random.uniform(size=(64, 64))
        yield dict(X=costly_function(X, n_ops),
                   y=np.random.randint(10, size=(1,)))