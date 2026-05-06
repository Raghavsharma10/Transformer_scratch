def timer(name, reservoir_type="uniform", *reservoir_args, **reservoir_kwargs):
    """
    Time-measuring context manager: the time spent in the wrapped block
    if measured and added to the named metric.
    """

    hmetric = get_or_create_histogram(name, reservoir_type, *reservoir_args, **reservoir_kwargs)

    t1 = time.time()
    yield
    t2 = time.time()
    hmetric.notify(t2 - t1)