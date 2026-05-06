def _parse(value, strict=True):
    """
    Preliminary duration value parser

    strict=True (by default) raises StrictnessError if either hours,
    minutes or seconds in duration value exceed allowed values
    """
    pattern = r'(?:(?P<hours>\d+):)?(?P<minutes>\d+):(?P<seconds>\d+)'
    match = re.match(pattern, value)
    if not match:
        raise ValueError('Invalid duration value: %s' % value)
    hours = safe_int(match.group('hours'))
    minutes = safe_int(match.group('minutes'))
    seconds = safe_int(match.group('seconds'))

    check_tuple((hours, minutes, seconds,), strict)

    return (hours, minutes, seconds,)