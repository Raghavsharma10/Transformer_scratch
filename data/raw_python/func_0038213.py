def from_str(duration):
    """Parse a duration string to a datetime.timedelta"""

    if duration in ("0", "+0", "-0"):
        return datetime.timedelta()

    pattern = re.compile('([\d\.]+)([a-zµμ]+)')
    total = 0
    sign = -1 if duration[0] == '-' else 1
    matches = pattern.findall(duration)

    if not len(matches):
        raise Exception("Invalid duration {}".format(duration))

    for (value, unit) in matches:
        if unit not in units:
            raise Exception(
                "Unknown unit {} in duration {}".format(unit, duration))
        try:
            total += float(value) * units[unit]
        except:
            raise Exception(
                "Invalid value {} in duration {}".format(value, duration))

    microseconds = total / _microsecond_size
    return datetime.timedelta(microseconds=sign * microseconds)