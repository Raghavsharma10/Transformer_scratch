def timestamp(datetime_obj):
    """Return Unix timestamp as float.

    The number of seconds that have elapsed since January 1, 1970.
    """
    start_of_time = datetime.datetime(1970, 1, 1)
    diff = datetime_obj - start_of_time
    return diff.total_seconds()