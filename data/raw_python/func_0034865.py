def contains_frame(data):
    """
    Read the frame length from the start of `data` and check if the data is
    long enough to contain the entire frame.
    """
    if len(data) < 2:
        return False

    b2 = struct.unpack('!B', data[1])[0]
    payload_len = b2 & 0x7F
    payload_start = 2

    if payload_len == 126:
        if len(data) > 4:
            payload_len = struct.unpack('!H', data[2:4])[0]

        payload_start = 4
    elif payload_len == 127:
        if len(data) > 12:
            payload_len = struct.unpack('!Q', data[4:12])[0]

        payload_start = 12

    return len(data) >= payload_len + payload_start