def loads(s, check_version=True):
    """
    Load VOEvent from bytes.

    This parses a VOEvent XML packet string, taking care of some subtleties.
    For Python 3 users, ``s`` should be a bytes object - see also
    http://lxml.de/FAQ.html,
    "Why can't lxml parse my XML from unicode strings?"
    (Python 2 users can stick with old-school ``str`` type if preferred)

    By default, will raise an exception if the VOEvent is not of version
    2.0. This can be disabled but voevent-parse routines are untested with
    other versions.

    Args:
        s (bytes): Bytes containing raw XML.
        check_version (bool): (Default=True) Checks that the VOEvent is of a
            supported schema version - currently only v2.0 is supported.
    Returns:
        :py:class:`Voevent`: Root-node of the  etree.
    Raises:
        ValueError: If passed a VOEvent of wrong schema version
            (i.e. schema 1.1)

    """
    # .. note::
    #
    # The namespace is removed from the root element tag to make
    #        objectify access work as expected,
    #        (see  :py:func:`._remove_root_tag_prefix`)
    #        so we must re-insert it when we want to conform to schema.
    v = objectify.fromstring(s)
    _remove_root_tag_prefix(v)

    if check_version:
        version = v.attrib['version']
        if not version == '2.0':
            raise ValueError('Unsupported VOEvent schema version:' + version)

    return v