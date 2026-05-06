def dumps(obj, preserve=False):
    """Stringifies a dict as toml

    :param obj: the object to be dumped into toml
    :param preserve: optional flag to preserve the inline table in result
    """
    f = StringIO()
    dump(obj, f, preserve)
    return f.getvalue()