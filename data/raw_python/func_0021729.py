def write_string(value, buff, byteorder='big'):
    """Write a string to a file-like object."""
    data = value.encode('utf-8')
    write_numeric(USHORT, len(data), buff, byteorder)
    buff.write(data)