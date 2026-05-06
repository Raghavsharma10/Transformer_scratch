def isoformat(dt):
    """Return an ISO-8601 formatted string from the provided datetime object"""
    if not isinstance(dt, datetime.datetime):
        raise TypeError("Must provide datetime.datetime object to isoformat")

    if dt.tzinfo is None:
        raise ValueError("naive datetime objects are not allowed beyond the library boundaries")

    return dt.isoformat().replace("+00:00", "Z")