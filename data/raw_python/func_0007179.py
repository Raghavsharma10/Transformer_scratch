def is_soon(dt, window):
    """Determines if time is going to happen in the next window seconds.

    :params dt: the time
    :params window: minimum seconds to remain to consider the time not soon

    :return: True if expiration is within the given duration
    """
    soon = (utcnow() + datetime.timedelta(seconds=window))
    return normalize_time(dt) <= soon