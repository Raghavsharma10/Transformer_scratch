def getShare(store, role, shareID):
    """
    Retrieve the accessible facet of an Item previously shared with
    L{shareItem}.

    This method is pending deprecation, and L{Role.getShare} should be
    preferred in new code.

    @param store: an axiom store (XXX must be the same as role.store)

    @param role: a L{Role}, the primary role for a user attempting to retrieve
    the given item.

    @return: a L{SharedProxy}.  This is a wrapper around the shared item which
    only exposes those interfaces explicitly allowed for the given role.

    @raise: L{NoSuchShare} if there is no item shared to the given role for the
    given shareID.
    """
    warnings.warn("Use Role.getShare() instead of sharing.getShare().",
                  PendingDeprecationWarning,
                  stacklevel=2)
    return role.getShare(shareID)