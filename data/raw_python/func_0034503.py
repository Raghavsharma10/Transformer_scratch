def _py2_crc16(value):
    """Calculate the CRC for the value in Python 2

    :param str value: The value to return for the CRC Checksum
    :rtype: int

    """
    crc = 0
    for byte in value:
        crc = ((crc << 8) & 0xffff) ^ \
              _CRC16_LOOKUP[((crc >> 8) ^ ord(byte)) & 0xff]
    return crc