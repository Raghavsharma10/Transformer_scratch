def uri_dump(uri, content, mode='wb', **kwargs):
    """
    Dumps the contents of a string/bytestring into a URI.
    See :func:`uri_open` for complete description of keyword parameters.

    :param str uri: URI to dump contents to
    :param str content: Contents to write to URI
    :param str mode: Either ``w``, or ``wb`` to write binary/text content respectiely
    """

    if 'r' in mode: raise ValueError('Read mode is not allowed for `uri_dump`.')

    with uri_open(uri, mode=mode, **kwargs) as f:
        f.write(content)
        f.flush()