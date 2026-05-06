def new_histogram(name, reservoir=None):
    """
    Build a new histogram metric with a given reservoir object
    If the reservoir is not provided, a uniform reservoir with the default size is used
    """

    if reservoir is None:
        reservoir = histogram.UniformReservoir(histogram.DEFAULT_UNIFORM_RESERVOIR_SIZE)

    return new_metric(name, histogram.Histogram, reservoir)