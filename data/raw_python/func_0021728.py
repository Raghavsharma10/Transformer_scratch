def read_string(buff, byteorder='big'):
    """Read a string from a file-like object."""
    length = read_numeric(USHORT, buff, byteorder)
    return buff.read(length).decode('utf-8')