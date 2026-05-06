def mock(config_or_spec=None, spec=None, strict=OMITTED):
    """Create 'empty' objects ('Mocks').

    Will create an empty unconfigured object, that you can pass
    around. All interactions (method calls) will be recorded and can be
    verified using :func:`verify` et.al.

    A plain `mock()` will be not `strict`, and thus all methods regardless
    of the arguments will return ``None``.

    .. note:: Technically all attributes will return an internal interface.
        Because of that a simple ``if mock().foo:`` will surprisingly pass.

    If you set strict to ``True``: ``mock(strict=True)`` all unexpected
    interactions will raise an error instead.

    You configure a mock using :func:`when`, :func:`when2` or :func:`expect`.
    You can also very conveniently just pass in a dict here::

        response = mock({'text': 'ok', 'raise_for_status': lambda: None})

    You can also create an empty Mock which is specced against a given
    `spec`: ``mock(requests.Response)``. These mock are by default strict,
    thus they raise if you want to stub a method, the spec does not implement.
    Mockito will also match the function signature.

    You can pre-configure a specced mock as well::

        response = mock({'json': lambda: {'status': 'Ok'}},
                        spec=requests.Response)

    Mocks are by default callable. Configure the callable behavior using
    `when`::

        dummy = mock()
        when(dummy).__call_(1).thenReturn(2)

    All other magic methods must be configured this way or they will raise an
    AttributeError.


    See :func:`verify` to verify your interactions after usage.

    """

    if type(config_or_spec) is dict:
        config = config_or_spec
    else:
        config = {}
        spec = config_or_spec

    if strict is OMITTED:
        strict = False if spec is None else True


    class Dummy(_Dummy):
        if spec:
            __class__ = spec  # make isinstance work

        def __getattr__(self, method_name):
            if strict:
                raise AttributeError(
                    "'Dummy' has no attribute %r configured" % method_name)
            return functools.partial(
                remembered_invocation_builder, theMock, method_name)

        def __repr__(self):
            name = 'Dummy'
            if spec:
                name += spec.__name__
            return "<%s id=%s>" % (name, id(self))


    # That's a tricky one: The object we will return is an *instance* of our
    # Dummy class, but the mock we register will point and patch the class.
    # T.i. so that magic methods (`__call__` etc.) can be configured.
    obj = Dummy()
    theMock = Mock(Dummy, strict=strict, spec=spec)

    for n, v in config.items():
        if inspect.isfunction(v):
            invocation.StubbedInvocation(theMock, n)(Ellipsis).thenAnswer(v)
        else:
            setattr(obj, n, v)

    mock_registry.register(obj, theMock)
    return obj