def transacted(func):
    """
    Return a callable which will invoke C{func} in a transaction using the
    C{store} attribute of the first parameter passed to it.  Typically this is
    used to create Item methods which are automatically run in a transaction.

    The attributes of the returned callable will resemble those of C{func} as
    closely as L{twisted.python.util.mergeFunctionMetadata} can make them.
    """
    def transactionified(item, *a, **kw):
        return item.store.transact(func, item, *a, **kw)
    return mergeFunctionMetadata(func, transactionified)