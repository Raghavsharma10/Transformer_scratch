def to_timedelta(value, strict=True):
    """
    converts duration string to timedelta

    strict=True (by default) raises StrictnessError if either hours,
    minutes or seconds in duration string exceed allowed values
    """
    if isinstance(value, int):
        return timedelta(seconds=value)  # assuming it's seconds
    elif isinstance(value, timedelta):
        return value
    elif isinstance(value, str):
        hours, minutes, seconds = _parse(value, strict)
    elif isinstance(value, tuple):
        check_tuple(value, strict)
        hours, minutes, seconds = value
    else:
        raise TypeError(
            'Value %s (type %s) not supported' % (
                value, type(value).__name__
            )
        )
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)