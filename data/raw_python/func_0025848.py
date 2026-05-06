def unframe(packet):
    """
    Strip leading DLE and trailing DLE/ETX from packet.

    :param packet: TSIP packet with leading DLE and trailing DLE/ETX.
    :type packet: Binary string.
    :return: TSIP packet with leading DLE and trailing DLE/ETX removed.
    :raise: ``ValueError`` if `packet` does not start with DLE and end in DLE/ETX.


    """

    if is_framed(packet):
        return packet.lstrip(CHR_DLE).rstrip(CHR_ETX).rstrip(CHR_DLE)
    else:
        raise ValueError('packet does not contain leading DLE and trailing DLE/ETX')