def to_iso8601(value, strict=True, force_int=True):
    """
    converts duration value to ISO8601 string
    accepts integers, hh:mm:ss or mm:ss strings, timedelta objects

    strict=True (by default) raises StrictnessError if either hours,
    minutes or seconds in duration string exceed allowed values
    """
    # split seconds to larger units
    # seconds = value.total_seconds()
    seconds = to_seconds(value, strict, force_int)

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    days, hours, minutes = map(int, (days, hours, minutes))
    seconds = round(seconds, 6)

    # build date
    date = ''
    if days:
        date = '%sD' % days

    # build time
    time = 'T'

    # hours
    bigger_exists = date or hours
    if bigger_exists:
        time += '{:02}H'.format(hours)

    # minutes
    bigger_exists = bigger_exists or minutes
    if bigger_exists:
        time += '{:02}M'.format(minutes)

    # seconds
    if isinstance(seconds, int) or force_int:
        seconds = '{:02}'.format(int(seconds))
    else:
        # 9 chars long w/leading 0, 6 digits after decimal
        seconds = '%09.6f' % seconds

    time += '{}S'.format(seconds)
    return 'P' + date + time