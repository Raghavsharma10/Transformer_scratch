def iso_datetime(timestamp=None):
    """ Convert UNIX timestamp to ISO datetime string.

        @param timestamp: UNIX epoch value (default: the current time).
        @return: Timestamp formatted as "YYYY-mm-dd HH:MM:SS".
    """
    if timestamp is None:
        timestamp = time.time()
    return datetime.datetime.fromtimestamp(timestamp).isoformat(' ')[:19]