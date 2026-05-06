def storeServiceSpecialCase(st, pups):
    """
    Adapt a store to L{IServiceCollection}.

    @param st: The L{Store} to adapt.
    @param pups: A list of L{IServiceCollection} powerups on C{st}.

    @return: An L{IServiceCollection} which has all of C{pups} as children.
    """
    if st.parent is not None:
        # If for some bizarre reason we're starting a substore's service, let's
        # just assume that its parent is running its upgraders, rather than
        # risk starting the upgrader run twice. (XXX: it *IS* possible to
        # figure out whether we need to or not, I just doubt this will ever
        # even happen in practice -- fix here if it does)
        return serviceSpecialCase(st, pups)
    if st._axiom_service is not None:
        # not new, don't add twice.
        return st._axiom_service

    collection = serviceSpecialCase(st, pups)

    st._upgradeService.setServiceParent(collection)

    if st.dbdir is not None:
        from axiom import batch
        batcher = batch.BatchProcessingControllerService(st)
        batcher.setServiceParent(collection)

    scheduler = iaxiom.IScheduler(st)
    # If it's an old database, we might get a SubScheduler instance.  It has no
    # setServiceParent method.
    setServiceParent = getattr(scheduler, 'setServiceParent', None)
    if setServiceParent is not None:
        setServiceParent(collection)

    return collection