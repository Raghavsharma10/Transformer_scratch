def parse_ts(ts):
    """
    parse timestamp.
    
    :param ts: timestamp in ISO8601 format
    :return: tbd!!!
    """
    # ISO8601 = '%Y-%m-%dT%H:%M:%SZ'
    # ISO8601_MS = '%Y-%m-%dT%H:%M:%S.%fZ'
    # RFC1123 = '%a, %d %b %Y %H:%M:%S %Z'
    dt = maya.parse(ts.strip())
    return dt.datetime(naive=True)