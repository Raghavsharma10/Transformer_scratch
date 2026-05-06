def parse_input(s):
    """Parse the given input and intelligently transform it into an absolute,
    non-naive, timezone-aware datetime object for the UTC timezone.

    The input can be specified as a millisecond-precision UTC timestamp (or
    delta against Epoch), with or without a terminating 'L'. Alternatively, the
    input can be specified as a human-readable delta string with unit-separated
    segments, like '24d6h4m500' (24 days, 6 hours, 4 minutes and 500ms), as
    long as the segments are in descending unit span order."""
    if isinstance(s, six.integer_types):
        s = str(s)
    elif not isinstance(s, six.string_types):
        raise ValueError(s)

    original = s

    if s[-1:] == 'L':
        s = s[:-1]

    sign = {'-': -1, '=': 0, '+': 1}.get(s[0], None)
    if sign is not None:
        s = s[1:]

    ts = 0
    for unit in _SORTED_UNITS:
        pos = s.find(unit[0])
        if pos == 0:
            raise ValueError(original)
        elif pos > 0:
            # If we find a unit letter, we're dealing with an offset. Default
            # to positive offset if a sign wasn't specified.
            if sign is None:
                sign = 1
            ts += int(s[:pos]) * __timedelta_millis(unit[1])
            s = s[min(len(s), pos + 1):]

    if s:
        ts += int(s)

    return date_from_utc_ts(ts) if not sign else \
        utc() + sign * delta(milliseconds=ts)