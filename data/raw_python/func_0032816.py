def _createLocalRouter(siteStore):
    """
    Create an L{IMessageRouter} provider for the default case, where no
    L{IMessageRouter} powerup is installed on the top-level store.

    It wraps a L{LocalMessageRouter} around the L{LoginSystem} installed on the
    given site store.

    If no L{LoginSystem} is present, this returns a null router which will
    simply log an error but not deliver the message anywhere, until this
    configuration error can be corrected.

    @rtype: L{IMessageRouter}
    """
    ls = siteStore.findUnique(LoginSystem, default=None)
    if ls is None:
        try:
            raise UnsatisfiedRequirement()
        except UnsatisfiedRequirement:
            log.err(Failure(),
                    "You have opened a substore from a site store with no "
                    "LoginSystem.  Message routing is disabled.")
        return _NullRouter()
    return LocalMessageRouter(ls)