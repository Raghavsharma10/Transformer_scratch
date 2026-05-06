def _errstr(value):
    """Returns the value str, truncated to MAX_ERROR_STR_LEN characters. If
    it's truncated, the returned value will have '...' on the end.
    """

    value = str(value) # We won't make the caller convert value to a string each time.
    if len(value) > MAX_ERROR_STR_LEN:
        return value[:MAX_ERROR_STR_LEN] + '...'
    else:
        return value