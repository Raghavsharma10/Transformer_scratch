def to_seconds(value, strict=True, force_int=True):
    """
    converts duration value to integer seconds

    strict=True (by default) raises StrictnessError if either hours,
    minutes or seconds in duration value exceed allowed values
    """
    if isinstance(value, int):
        return value  # assuming it's seconds
    elif isinstance(value, timedelta):
        seconds = value.total_seconds()
        if force_int:
            seconds = int(round(seconds))
        return seconds
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

    if not (hours or minutes or seconds):
        raise ValueError('No hours, minutes or seconds found')

    result = hours*3600 + minutes*60 + seconds
    return result