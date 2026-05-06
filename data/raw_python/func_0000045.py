def when2(fn, *args, **kwargs):
    """Stub a function call with the given arguments

    Exposes a more pythonic interface than :func:`when`. See :func:`when` for
    more documentation.

    Returns `AnswerSelector` interface which exposes `thenReturn`,
    `thenRaise`, and `thenAnswer` as usual. Always `strict`.

    Usage::

        # Given `dog` is an instance of a `Dog`
        when2(dog.bark, 'Miau').thenReturn('Wuff')

    .. note:: You must :func:`unstub` after stubbing, or use `with`
        statement.

    """
    obj, name = get_obj_attr_tuple(fn)
    theMock = _get_mock(obj, strict=True)
    return invocation.StubbedInvocation(theMock, name)(*args, **kwargs)