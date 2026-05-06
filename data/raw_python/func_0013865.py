def delta_ps(prev, curr, counters):
    """ calculate the delta per second of one counter

    formula: (curr - prev) / delta_time
    :param prev: previous resource
    :param curr: current resource
    :param counters: the counter to do delta and per second, one only
    :return: value, NaN if invalid.
    """
    counter = get_counter(counters)

    pv = getattr(prev, counter)
    cv = getattr(curr, counter)
    return minus(cv, pv)