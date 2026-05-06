def _quoted(value):
    """Return a single-quoted and escaped (percent-encoded) version of value

    This function will also perform transforms of known data types to a representation
    that will be handled by Device Cloud.  For instance, datetime objects will be
    converted to ISO8601.

    """
    if isinstance(value, datetime.datetime):
        value = isoformat(to_none_or_dt(value))
    else:
        value = str(value)

    return "'{}'".format(value)