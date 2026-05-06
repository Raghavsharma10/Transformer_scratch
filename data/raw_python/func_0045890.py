def isstring(value):
    """Report whether the given value is a byte or unicode string."""
    classes = (str, bytes) if pyutils.PY3 else basestring  # noqa: F821
    return isinstance(value, classes)