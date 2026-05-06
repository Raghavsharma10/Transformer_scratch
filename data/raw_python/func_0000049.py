def verifyZeroInteractions(*objs):
    """Verify that no methods have been called on given objs.

    Note that strict mocks usually throw early on unexpected, unstubbed
    invocations. Partial mocks ('monkeypatched' objects or modules) do not
    support this functionality at all, bc only for the stubbed invocations
    the actual usage gets recorded. So this function is of limited use,
    nowadays.

    """
    for obj in objs:
        theMock = _get_mock_or_raise(obj)

        if len(theMock.invocations) > 0:
            raise VerificationError(
                "\nUnwanted interaction: %s" % theMock.invocations[0])