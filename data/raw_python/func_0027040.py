def utime_delta(days=0, hours=0, minutes=0, seconds=0):
    """Gets time delta in microseconds.

    Note: Do NOT use this function without keyword arguments.
    It will become much-much harder to add extra time ranges later if positional arguments are used.

    """
    return (days * DAY) + (hours * HOUR) + (minutes * MINUTE) + (seconds * SECOND)