def decode_format_timestamp(timestamp):
    """Convert unix timestamp (millis) into date & time we use in logs output.

    :param timestamp: unix timestamp in millis
    :return: date, time in UTC
    """
    dt = maya.MayaDT(timestamp / 1000).datetime(naive=True)
    return dt.strftime('%Y-%m-%d'), dt.strftime('%H:%M:%S')