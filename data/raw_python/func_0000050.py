def verifyNoUnwantedInteractions(*objs):
    """Verifies that expectations set via `expect` are met

    E.g.::

        expect(os.path, times=1).exists(...).thenReturn(True)
        os.path('/foo')
        verifyNoUnwantedInteractions(os.path)  # ok, called once

    If you leave out the argument *all* registered objects will
    be checked.

    .. note:: **DANGERZONE**: If you did not :func:`unstub` correctly,
        it is possible that old registered mocks, from other tests
        leak.

    See related :func:`expect`
    """

    if objs:
        theMocks = map(_get_mock_or_raise, objs)
    else:
        theMocks = mock_registry.get_registered_mocks()

    for mock in theMocks:
        for i in mock.stubbed_invocations:
            i.verify()