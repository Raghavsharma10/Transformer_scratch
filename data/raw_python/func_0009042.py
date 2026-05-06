def timestamp_to_datetime(timestamp):
    """Convert an ARF timestamp to a datetime.datetime object (naive local time)"""
    from datetime import datetime, timedelta
    obj = datetime.fromtimestamp(timestamp[0])
    return obj + timedelta(microseconds=int(timestamp[1]))