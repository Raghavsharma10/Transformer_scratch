def to_unicode(obj, encoding='utf-8', fallback='latin1', **decode_args):
    r"""
    Returns a ``unicode`` of ``obj``, decoding using ``encoding`` if necessary.
    If decoding fails, the ``fallback`` encoding (default ``latin1``) is used.

    Examples::

        >>> r(to_unicode(b'\xe1\x88\xb4'))
        u'\u1234'
        >>> r(to_unicode(b'\xff'))
        u'\xff'
        >>> r(to_unicode(u'\u1234'))
        u'\u1234'
        >>> r(to_unicode(Exception(u'\u1234')))
        u'\u1234'
        >>> r(to_unicode([42]))
        u'[42]'

    See source code for detailed semantics.
    """

    # Note: on py3, the `bytes` type defines an unhelpful "__str__" function,
    # so we need to do this check (see comments in ``to_str``).
    if not isinstance(obj, binary_type):
        if isinstance(obj, text_type) or hasattr(obj, text_type_magicmethod):
            return text_type(obj)

        obj_str = binary_type(obj)
    else:
        obj_str = obj

    try:
        return text_type(obj_str, encoding, **decode_args)
    except UnicodeDecodeError:
        return text_type(obj_str, fallback, **decode_args)