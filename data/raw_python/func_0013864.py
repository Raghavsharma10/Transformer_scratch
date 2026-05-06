def utilization(prev, curr, counters):
    """ calculate the utilization

    delta_busy = curr.busy - prev.busy
    delta_idle = curr.idle - prev.idle
    utilization = delta_busy / (delta_busy + delta_idle)

    :param prev: previous resource
    :param curr: current resource
    :param counters: list of two, busy ticks and idle ticks
    :return: value, NaN if invalid.
    """
    busy_prop, idle_prop = counters

    pb = getattr(prev, busy_prop)
    pi = getattr(prev, idle_prop)

    cb = getattr(curr, busy_prop)
    ci = getattr(curr, idle_prop)

    db = minus(cb, pb)
    di = minus(ci, pi)

    return mul(div(db, add(db, di)), 100)