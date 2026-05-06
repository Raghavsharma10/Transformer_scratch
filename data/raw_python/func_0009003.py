def surrogate_escape(error):
    """
    Simulate the Python 3 ``surrogateescape`` handler, but for Python 2 only.
    """
    chars = error.object[error.start:error.end]
    assert len(chars) == 1
    val = ord(chars)
    val += 0xdc00
    return __builtin__.unichr(val), error.end