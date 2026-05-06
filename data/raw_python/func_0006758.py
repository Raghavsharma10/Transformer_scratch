def stream_to_packet(data):
    """
    Chop a stream of data into MODBUS packets.

    :param data: stream of data
    :returns: a tuple of the data that is a packet with the remaining
        data, or ``None``
    """
    if len(data) < 6:
        return None

    # unpack the length
    pktlen = struct.unpack(">H", data[4:6])[0] + 6
    if (len(data) < pktlen):
        return None

    return (data[:pktlen], data[pktlen:])