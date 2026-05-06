def getDescriptor(klass, **kw):
    """
    Automatically fills bLength and bDescriptorType.
    """
    # XXX: ctypes Structure.__init__ ignores arguments which do not exist
    # as structure fields. So check it.
    # This is annoying, but not doing it is a huge waste of time for the
    # developer.
    empty = klass()
    assert hasattr(empty, 'bLength')
    assert hasattr(empty, 'bDescriptorType')
    unknown = [x for x in kw if not hasattr(empty, x)]
    if unknown:
        raise TypeError('Unknown fields %r' % (unknown, ))
    # XXX: not very pythonic...
    return klass(
        bLength=ctypes.sizeof(klass),
        # pylint: disable=protected-access
        bDescriptorType=klass._bDescriptorType,
        # pylint: enable=protected-access
        **kw
    )