def verifyStubbedInvocationsAreUsed(*objs):
    """Ensure stubs are actually used.

    This functions just ensures that stubbed methods are actually used. Its
    purpose is to detect interface changes after refactorings. It is meant
    to be invoked usually without arguments just before :func:`unstub`.

    """
    if objs:
        theMocks = map(_get_mock_or_raise, objs)
    else:
        theMocks = mock_registry.get_registered_mocks()


    for mock in theMocks:
        for i in mock.stubbed_invocations:
            if not i.allow_zero_invocations and i.used < len(i.answers):
                raise VerificationError("\nUnused stub: %s" % i)