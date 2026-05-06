def datetime_to_timestamp(time: datetime) -> Timestamp:
    """
    Convert datetime to protobuf.timestamp.

    :param time: time
    :type time: ~datetime.datetime
    :return: protobuf.timestamp
    :rtype: ~google.protobuf.timestamp_pb2.Timestamp
    """
    protime = Timestamp()
    protime.FromDatetime(time)
    return protime