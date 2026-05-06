def read_numeric(fmt, buff, byteorder='big'):
    """Read a numeric value from a file-like object."""
    try:
        fmt = fmt[byteorder]
        return fmt.unpack(buff.read(fmt.size))[0]
    except StructError:
        return 0
    except KeyError as exc:
        raise ValueError('Invalid byte order') from exc