def getAuthenticatedRole(store):
    """
    Get the base 'Authenticated' role for this store, which is the role that is
    given to every user who is explicitly identified by a non-anonymous
    username.
    """
    def tx():
        def addToEveryone(newAuthenticatedRole):
            newAuthenticatedRole.becomeMemberOf(getEveryoneRole(store))
            return newAuthenticatedRole
        return store.findOrCreate(Role, addToEveryone, externalID=u'Authenticated')
    return store.transact(tx)