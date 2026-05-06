def patch(fn, attr_or_replacement, replacement=None):
    """Patch/Replace a function.

    This is really like monkeypatching, but *note* that all interactions
    will be recorded and can be verified. That is, using `patch` you stay in
    the domain of mockito.

    Two ways to call this. Either::

        patch(os.path.exists, lambda str: True)  # two arguments
        # OR
        patch(os.path, 'exists', lambda str: True)  # three arguments

    If called with three arguments, the mode is *not* strict to allow *adding*
    methods. If called with two arguments, mode is always `strict`.

    .. note:: You must :func:`unstub` after stubbing, or use `with`
        statement.

    """
    if replacement is None:
        replacement = attr_or_replacement
        return when2(fn, Ellipsis).thenAnswer(replacement)
    else:
        obj, name = fn, attr_or_replacement
        theMock = _get_mock(obj, strict=True)
        return invocation.StubbedInvocation(
            theMock, name, strict=False)(Ellipsis).thenAnswer(replacement)