def to_str(obj, encoding='utf-8', **encode_args):
    r"""
    Returns a ``str`` of ``obj``, encoding using ``encoding`` if necessary. For
    example::

        >>> some_str = b"\xff"
        >>> some_unicode = u"\u1234"
        >>> some_exception = Exception(u'Error: ' + some_unicode)
        >>> r(to_str(some_str))
        b'\xff'
        >>> r(to_str(some_unicode))
        b'\xe1\x88\xb4'
        >>> r(to_str(some_exception))
        b'Error: \xe1\x88\xb4'
        >>> r(to_str([42]))
        b'[42]'

    See source code for detailed semantics.
    """
    # Note: On py3, ``b'x'.__str__()`` returns ``"b'x'"``, so we need to do the
    # explicit check first.
    if isinstance(obj, binary_type):
        return obj

    # We coerce to unicode if '__unicode__' is available because there is no
    # way to specify encoding when calling ``str(obj)``, so, eg,
    # ``str(Exception(u'\u1234'))`` will explode.
    if isinstance(obj, text_type) or hasattr(obj, text_type_magicmethod):
        # Note: unicode(u'foo') is O(1) (by experimentation)
        return text_type(obj).encode(encoding, **encode_args)

    return binary_type(obj)