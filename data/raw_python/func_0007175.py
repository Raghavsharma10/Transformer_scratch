def advance_time_delta(timedelta):
    """Advance overridden time using a datetime.timedelta."""
    assert(utcnow.override_time is not None)
    try:
        for dt in utcnow.override_time:
            dt += timedelta
    except TypeError:
        utcnow.override_time += timedelta