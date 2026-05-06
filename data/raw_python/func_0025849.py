def stuff(packet):
    """
    Add byte stuffing to TSIP packet.
    :param packet: TSIP packet with byte stuffing. The packet must already
        have been stripped or `ValueError` will be raised.
    :type packet: Binary string.
    :return: Packet with byte stuffing.

    """

    if is_framed(packet):
        raise ValueError('packet contains leading DLE and trailing DLE/ETX')
    else:
        return packet.replace(CHR_DLE, CHR_DLE + CHR_DLE)