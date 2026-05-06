def when(obj, strict=None):
    """Central interface to stub functions on a given `obj`

    `obj` should be a module, a class or an instance of a class; it can be
    a Dummy you created with :func:`mock`. ``when`` exposes a fluent interface
    where you configure a stub in three steps::

        when(<obj>).<method_name>(<args>).thenReturn(<value>)

    Compared to simple *patching*, stubbing in mockito requires you to specify
    conrete `args` for which the stub will answer with a concrete `<value>`.
    All invocations that do not match this specific call signature will be
    rejected. They usually throw at call time.

    Stubbing in mockito's sense thus means not only to get rid of unwanted
    side effects, but effectively to turn function calls into constants.

    E.g.::

        # Given ``dog`` is an instance of a ``Dog``
        when(dog).bark('Grrr').thenReturn('Wuff')
        when(dog).bark('Miau').thenRaise(TypeError())

        # With this configuration set up:
        assert dog.bark('Grrr') == 'Wuff'
        dog.bark('Miau')  # will throw TypeError
        dog.bark('Wuff')  # will throw unwanted interaction

    Stubbing can effectively be used as monkeypatching; usage shown with
    the `with` context managing::

        with when(os.path).exists('/foo').thenReturn(True):
            ...

    Most of the time verifying your interactions is not necessary, because
    your code under tests implicitly verifies the return value by evaluating
    it. See :func:`verify` if you need to, see also :func:`expect` to setup
    expected call counts up front.

    If your function is pure side effect and does not return something, you
    can omit the specific answer. The default then is `None`::

        when(manager).do_work()

    `when` verifies the method name, the expected argument signature, and the
    actual, factual arguments your code under test uses against the original
    object and its function so its easier to spot changing interfaces.

    Sometimes it's tedious to spell out all arguments::

        from mockito import ANY, ARGS, KWARGS
        when(requests).get('http://example.com/', **KWARGS).thenReturn(...)
        when(os.path).exists(ANY)
        when(os.path).exists(ANY(str))

    .. note:: You must :func:`unstub` after stubbing, or use `with`
        statement.

    Set ``strict=False`` to bypass the function signature checks.

    See related :func:`when2` which has a more pythonic interface.

    """

    if isinstance(obj, str):
        obj = get_obj(obj)

    if strict is None:
        strict = True
    theMock = _get_mock(obj, strict=strict)

    class When(object):
        def __getattr__(self, method_name):
            return invocation.StubbedInvocation(
                theMock, method_name, strict=strict)

    return When()