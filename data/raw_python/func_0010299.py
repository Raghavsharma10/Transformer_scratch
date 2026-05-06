def iso8601_to_dt(iso8601):
    """Given an ISO8601 string as returned by Device Cloud, convert to a datetime object"""
    # We could just use arrow.get() but that is more permissive than we actually want.
    # Internal (but still public) to arrow is the actual parser where we can be
    # a bit more specific
    parser = DateTimeParser()
    try:
        arrow_dt = arrow.Arrow.fromdatetime(parser.parse_iso(iso8601))
        return arrow_dt.to('utc').datetime
    except ParserError as pe:
        raise ValueError("Provided was not a valid ISO8601 string: %r" % pe)