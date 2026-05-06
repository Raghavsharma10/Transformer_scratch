def get_time_from_rfc3339(rfc3339):
    """
    return time tuple from an RFC 3339-formatted time string

    :param rfc3339: str, time in RFC 3339 format
    :return: float, seconds since the Epoch
    """

    try:
        # py 3

        dt = dateutil.parser.parse(rfc3339, ignoretz=False)
        return dt.timestamp()
    except NameError:
        # py 2

        # Decode the RFC 3339 date with no fractional seconds (the
        # format Origin provides). Note that this will fail to parse
        # valid ISO8601 timestamps not in this exact format.
        time_tuple = strptime(rfc3339, '%Y-%m-%dT%H:%M:%SZ')
        return timegm(time_tuple)