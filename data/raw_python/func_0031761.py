def get_or_create_histogram(name, reservoir_type, *reservoir_args, **reservoir_kwargs):
    """
    Will return a histogram matching the given parameters or raise
    DuplicateMetricError if it can't be created due to a name collision
    with another histogram with different parameters.
    """
    reservoir = new_reservoir(reservoir_type, *reservoir_args, **reservoir_kwargs)

    try:
        hmetric = new_histogram(name, reservoir)
    except DuplicateMetricError:
        hmetric = metric(name)
        if not isinstance(hmetric, histogram.Histogram):
            raise DuplicateMetricError(
                "Metric {!r} already exists of type {!r}".format(name, type(hmetric).__name__))

        if not hmetric.reservoir.same_kind(reservoir):
            raise DuplicateMetricError(
                "Metric {!r} already exists with a different reservoir: {}".format(name, hmetric.reservoir))

    return hmetric