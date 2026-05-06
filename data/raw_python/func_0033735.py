def b64_from(val):
    """Returns base64 encoded bytes for a given int/long/bytes value.

    :param int|long|bytes val:
    :rtype: bytes|str
    """
    if isinstance(val, integer_types):
        val = int_to_bytes(val)
    return b64encode(val).decode('ascii')