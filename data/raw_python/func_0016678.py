def datetime_to_timestamp(dt):
    """Convert timezone-aware `datetime` to POSIX timestamp and
    return seconds since UNIX epoch.

    Note: similar to `datetime.timestamp()` in Python 3.3+.
    """

    epoch = datetime.utcfromtimestamp(0).replace(tzinfo=UTC)
    return (dt - epoch).total_seconds()