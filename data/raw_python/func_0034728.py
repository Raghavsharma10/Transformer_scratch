def encode_date_optional_time(obj):
    """
    ISO encode timezone-aware datetimes
    """
    if isinstance(obj, datetime.datetime):
        return timezone("UTC").normalize(obj.astimezone(timezone("UTC"))).strftime('%Y-%m-%dT%H:%M:%SZ')
    raise TypeError("{0} is not JSON serializable".format(repr(obj)))