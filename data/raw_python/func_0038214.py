def to_str(delta, extended=False):
    """Format a datetime.timedelta to a duration string"""

    total_seconds = delta.total_seconds()
    sign = "-" if total_seconds < 0 else ""
    nanoseconds = abs(total_seconds * _second_size)

    if total_seconds < 1:
        result_str = _to_str_small(nanoseconds, extended)
    else:
        result_str = _to_str_large(nanoseconds, extended)

    return "{}{}".format(sign, result_str)