def with_histogram(name, reservoir_type="uniform", *reservoir_args, **reservoir_kwargs):
    """
    Time-measuring decorator: the time spent in the wrapped function is measured
    and added to the named metric.
    metric_args and metric_kwargs are passed to new_histogram()
    """

    hmetric = get_or_create_histogram(name, reservoir_type, *reservoir_args, **reservoir_kwargs)

    def wrapper(f):

        @functools.wraps(f)
        def fun(*args, **kwargs):
            t1 = time.time()
            res = f(*args, **kwargs)
            t2 = time.time()

            hmetric.notify(t2-t1)
            return res

        return fun

    return wrapper