def native_path(path):  # pragma: no cover
    """
    Always return a native path, that is unicode on Python 3 and bytestring on
    Python 2.

    Taken `from Django <http://bit.ly/1r3gogZ>`_.
    """
    if PY2 and not isinstance(path, bytes):
        return path.encode(fs_encoding)
    return path