def spy(object):
    """Spy an object.

    Spying means that all functions will behave as before, so they will
    be side effects, but the interactions can be verified afterwards.

    Returns Dummy-like, almost empty object as proxy to `object`.

    The *returned* object must be injected and used by the code under test;
    after that all interactions can be verified as usual.
    T.i. the original object **will not be patched**, and has no further
    knowledge as before.

    E.g.::

        import time
        time = spy(time)
        # inject time
        do_work(..., time)
        verify(time).time()

    """
    if inspect.isclass(object) or inspect.ismodule(object):
        class_ = None
    else:
        class_ = object.__class__

    class Spy(_Dummy):
        if class_:
            __class__ = class_

        def __getattr__(self, method_name):
            return RememberedProxyInvocation(theMock, method_name)

        def __repr__(self):
            name = 'Spied'
            if class_:
                name += class_.__name__
            return "<%s id=%s>" % (name, id(self))


    obj = Spy()
    theMock = Mock(obj, strict=True, spec=object)

    mock_registry.register(obj, theMock)
    return obj