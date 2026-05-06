def to_tuple(value, strict=True, force_int=True):
    """
    converts duration value to tuple of integers

    strict=True (by default) raises StrictnessError if either hours,
    minutes or seconds in duration value exceed allowed values
    """
    if isinstance(value, int):
        seconds = value
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
    elif isinstance(value, str):
        hours, minutes, seconds = _fix_tuple(
            _parse(value, strict)
        )
    elif isinstance(value, tuple):
        check_tuple(value, strict)
        hours, minutes, seconds = _fix_tuple(value)
    elif isinstance(value, timedelta):
        seconds = value.total_seconds()
        if force_int:
            seconds = int(round(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)

    return (hours, minutes, seconds,)