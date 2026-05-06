def dump(obj, f, preserve=False):
    """Write dict object into file

    :param obj: the object to be dumped into toml
    :param f: the file object
    :param preserve: optional flag to preserve the inline table in result
    """
    if not f.write:
        raise TypeError('You can only dump an object into a file object')
    encoder = Encoder(f, preserve=preserve)
    return encoder.write_dict(obj)