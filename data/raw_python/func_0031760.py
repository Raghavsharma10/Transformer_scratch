def new_reservoir(reservoir_type='uniform', *reservoir_args, **reservoir_kwargs):
    """
    Build a new reservoir
    """

    try:
        reservoir_cls = RESERVOIR_TYPES[reservoir_type]
    except KeyError:
        raise InvalidMetricError("Unknown reservoir type: {}".format(reservoir_type))

    return reservoir_cls(*reservoir_args, **reservoir_kwargs)