def getAccountRole(store, accountNames):
    """
    Retrieve the first Role in the given store which corresponds an account
    name in C{accountNames}.

    Note: the implementation currently ignores all of the values in
    C{accountNames} except for the first.

    @param accountNames: A C{list} of two-tuples of account local parts and
        domains.

    @raise ValueError: If C{accountNames} is empty.

    @rtype: L{Role}
    """
    for (localpart, domain) in accountNames:
        return getPrimaryRole(store, u'%s@%s' % (localpart, domain),
                              createIfNotFound=True)
    raise ValueError("Cannot get named role for unnamed account.")