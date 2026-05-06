def getLoginMethods(store, protocol=None):
    """
    Retrieve L{LoginMethod} items from store C{store}, optionally constraining
    them by protocol
    """
    if protocol is not None:
        comp = OR(LoginMethod.protocol == u'*',
                  LoginMethod.protocol == protocol)
    else:
        comp = None
    return store.query(LoginMethod, comp)