def with_meter(name, tick_interval=meter.DEFAULT_TICK_INTERVAL):
    """
    Call-counting decorator: each time the wrapped function is called
    the named meter is incremented by one.
    metric_args and metric_kwargs are passed to new_meter()
    """

    try:
        mmetric = new_meter(name, tick_interval)
    except DuplicateMetricError as e:
        mmetric = metric(name)

        if not isinstance(mmetric, meter.Meter):
            raise DuplicateMetricError("Metric {!r} already exists of type {}".format(name, type(mmetric).__name__))

        if tick_interval != mmetric.tick_interval:
            raise DuplicateMetricError("Metric {!r} already exists: {}".format(name, mmetric))

    def wrapper(f):

        @functools.wraps(f)
        def fun(*args, **kwargs):
            res = f(*args, **kwargs)

            mmetric.notify(1)
            return res

        return fun

    return wrapper