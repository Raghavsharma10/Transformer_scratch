def new_histogram_with_implicit_reservoir(name, reservoir_type='uniform', *reservoir_args, **reservoir_kwargs):
    """
    Build a new histogram metric and a reservoir from the given parameters
    """

    reservoir = new_reservoir(reservoir_type, *reservoir_args, **reservoir_kwargs)
    return new_histogram(name, reservoir)