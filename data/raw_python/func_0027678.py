def storeBatchServiceSpecialCase(st, pups):
    """
    Adapt a L{Store} to L{IBatchService}.

    If C{st} is a substore, return a simple wrapper that delegates to the site
    store's L{IBatchService} powerup.  Return C{None} if C{st} has no
    L{BatchProcessingControllerService}.
    """
    if st.parent is not None:
        try:
            return _SubStoreBatchChannel(st)
        except TypeError:
            return None
    storeService = service.IService(st)
    try:
        return storeService.getServiceNamed("Batch Processing Controller")
    except KeyError:
        return None