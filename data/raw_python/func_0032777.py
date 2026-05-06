def asAccessibleTo(role, query):
    """
    Return an iterable which yields the shared proxies that are available to
    the given role, from the given query.

    This method is pending deprecation, and L{Role.asAccessibleTo} should be
    preferred in new code.

    @param role: The role to retrieve L{SharedProxy}s for.

    @param query: An Axiom query describing the Items to retrieve, which this
    role can access.
    @type query: an L{iaxiom.IQuery} provider.
    """
    warnings.warn(
        "Use Role.asAccessibleTo() instead of sharing.asAccessibleTo().",
        PendingDeprecationWarning,
        stacklevel=2)
    return role.asAccessibleTo(query)