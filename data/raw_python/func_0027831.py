def getAccountNames(store, protocol=None):
    """
    Retrieve account name information about the given database.

    @param store: An Axiom Store representing a user account.  It must
    have been opened through the store which contains its account
    information.

    @return: A generator of two-tuples of (username, domain) which
    refer to the given store.
    """
    return ((meth.localpart, meth.domain) for meth
                in getLoginMethods(store, protocol))