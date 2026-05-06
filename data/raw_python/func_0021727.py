def write_numeric(fmt, value, buff, byteorder='big'):
    """Write a numeric value to a file-like object."""
    try:
        buff.write(fmt[byteorder].pack(value))
    except KeyError as exc:
        raise ValueError('Invalid byte order') from exc