def uri_read(*args, **kwargs):
    """
    Reads the contents of a URI into a string or bytestring.
    See :func:`uri_open` for complete description of keyword parameters.

    :returns: Contents of URI
    :rtype: str, bytes
    """

    with uri_open(*args, **kwargs) as f:
        content = f.read()
    return content