def bytes_to_string(raw):
    """Convert bytes to string."""
    ret = bytes()
    for byte in raw:
        if byte == 0x00:
            return ret.decode("utf-8")
        ret += bytes([byte])
    return ret.decode("utf-8")