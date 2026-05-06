def shareItem(sharedItem, toRole=None, toName=None, shareID=None,
              interfaces=ALL_IMPLEMENTED):
    """
    Share an item with a given role.  This provides a way to expose items to
    users for later retrieval with L{Role.getShare}.

    This API is slated for deprecation.  Prefer L{Role.shareItem} in new code.

    @param sharedItem: an item to be shared.

    @param toRole: a L{Role} instance which represents the group that has
    access to the given item.  May not be specified if toName is also
    specified.

    @param toName: a unicode string which uniquely identifies a L{Role} in the
    same store as the sharedItem.

    @param shareID: a unicode string.  If provided, specify the ID under which
    the shared item will be shared.

    @param interfaces: a list of Interface objects which specify the methods
    and attributes accessible to C{toRole} on C{sharedItem}.

    @return: a L{Share} which records the ability of the given role to access
    the given item.
    """
    warnings.warn("Use Role.shareItem() instead of sharing.shareItem().",
                  PendingDeprecationWarning,
                  stacklevel=2)
    if toRole is None:
        if toName is not None:
            toRole = getPrimaryRole(sharedItem.store, toName, True)
        else:
            toRole = getEveryoneRole(sharedItem.store)
    return toRole.shareItem(sharedItem, shareID, interfaces)