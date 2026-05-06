def _missingExtraCheck(given, required, extraException, missingException):
    """
    If the L{sets<set>} C{required} and C{given} do not contain the same
    elements raise an exception describing how they are different.

    @param given: The L{set} of elements that was actually given.
    @param required: The L{set} of elements that must be given.

    @param extraException: An exception to raise if there are elements in
        C{given} that are not in C{required}.
    @param missingException: An exception to raise if there are elements in
        C{required} that are not in C{given}.

    @return: C{None}
    """
    extra = given - required
    if extra:
        raise extraException(extra)

    missing = required - given
    if missing:
        raise missingException(missing)