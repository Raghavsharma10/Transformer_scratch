def expect(obj, strict=None,
           times=None, atleast=None, atmost=None, between=None):
    """Stub a function call, and set up an expected call count.

    Usage::

        # Given `dog` is an instance of a `Dog`
        expect(dog, times=1).bark('Wuff').thenReturn('Miau')
        dog.bark('Wuff')
        dog.bark('Wuff')  # will throw at call time: too many invocations

        # maybe if you need to ensure that `dog.bark()` was called at all
        verifyNoUnwantedInteractions()

    .. note:: You must :func:`unstub` after stubbing, or use `with`
        statement.

    See :func:`when`, :func:`when2`, :func:`verifyNoUnwantedInteractions`

    """
    if strict is None:
        strict = True
    theMock = _get_mock(obj, strict=strict)

    verification_fn = _get_wanted_verification(
        times=times, atleast=atleast, atmost=atmost, between=between)

    class Expect(object):
        def __getattr__(self, method_name):
            return invocation.StubbedInvocation(
                theMock, method_name, verification=verification_fn,
                strict=strict)

    return Expect()