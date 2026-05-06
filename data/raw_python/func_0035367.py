def human_duration(time1, time2=None, precision=0, short=False):
    """ Return a human-readable representation of a time delta.

        @param time1: Relative time value.
        @param time2: Time base (C{None} for now; 0 for a duration in C{time1}).
        @param precision: How many time units to return (0 = all).
        @param short: Use abbreviations, and right-justify the result to always the same length.
        @return: Formatted duration.
    """
    if time2 is None:
        time2 = time.time()

    duration = (time1 or 0) - time2
    direction = (
        " ago" if duration < 0 else
        ("+now" if short else " from now") if time2 else ""
    )
    duration = abs(duration)
    parts = [
        ("weeks", duration // (7*86400)),
        ("days", duration // 86400 % 7),
        ("hours", duration // 3600 % 24),
        ("mins", duration // 60 % 60),
        ("secs", duration % 60),
    ]

    # Kill leading zero parts
    while len(parts) > 1 and parts[0][1] == 0:
        parts = parts[1:]

    # Limit to # of parts given by precision
    if precision:
        parts = parts[:precision]

    numfmt = ("%d", "%d"), ("%4d", "%2d")
    fmt = "%1.1s" if short else " %s"
    sep = " " if short else ", "
    result = sep.join((numfmt[bool(short)][bool(idx)] + fmt) % (val, key[:-1] if val == 1 else key)
        for idx, (key, val) in enumerate(parts)
        if val #or (short and precision)
    ) + direction

    if not time1:
        result = "never" if time2 else "N/A"

    if precision and short:
        return result.rjust(1 + precision*4 + (4 if time2 else 0))
    else:
        return result