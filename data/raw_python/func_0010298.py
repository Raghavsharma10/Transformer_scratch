def conditional_write(strm, fmt, value, *args, **kwargs):
    """Write to stream using fmt and value if value is not None"""
    if value is not None:
        strm.write(fmt.format(value, *args, **kwargs))