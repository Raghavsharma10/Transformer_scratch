def getPrimaryRole(store, primaryRoleName, createIfNotFound=False):
    """
    Get Role object corresponding to an identifier name.  If the role name
    passed is the empty string, it is assumed that the user is not
    authenticated, and the 'Everybody' role is primary.  If the role name
    passed is non-empty, but has no corresponding role, the 'Authenticated'
    role - which is a member of 'Everybody' - is primary.  Finally, a specific
    role can be primary if one exists for the user's given credentials, that
    will automatically always be a member of 'Authenticated', and by extension,
    of 'Everybody'.

    @param primaryRoleName: a unicode string identifying the role to be
    retrieved.  This corresponds to L{Role}'s externalID attribute.

    @param createIfNotFound: a boolean.  If True, create a role for the given
    primary role name if no exact match is found.  The default, False, will
    instead retrieve the 'nearest match' role, which can be Authenticated or
    Everybody depending on whether the user is logged in or not.

    @return: a L{Role}.
    """
    if not primaryRoleName:
        return getEveryoneRole(store)
    ff = store.findUnique(Role, Role.externalID == primaryRoleName, default=None)
    if ff is not None:
        return ff
    authRole = getAuthenticatedRole(store)
    if createIfNotFound:
        role = Role(store=store,
                    externalID=primaryRoleName)
        role.becomeMemberOf(authRole)
        return role
    return authRole