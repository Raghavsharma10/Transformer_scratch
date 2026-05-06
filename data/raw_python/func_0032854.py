def _webTranslator(store, fallback):
    """
    Discover a web translator based on an Axiom store and a specified default.
    Prefer the specified default.

    This is an implementation detail of various initializers in this module
    which require an L{IWebTranslator} provider.  Some of those initializers
    did not previously require a webTranslator, so this function will issue a
    L{UserWarning} if no L{IWebTranslator} powerup exists for the given store
    and no fallback is provided.

    @param store: an L{axiom.store.Store}
    @param fallback: a provider of L{IWebTranslator}, or None

    @return: 'fallback', if it is provided, or the L{IWebTranslator} powerup on
    'store'.
    """
    if fallback is None:
        fallback = IWebTranslator(store, None)
        if fallback is None:
            warnings.warn(
                "No IWebTranslator plugin when creating Scrolltable - broken "
                "configuration, now deprecated!  Try passing webTranslator "
                "keyword argument.", category=DeprecationWarning,
                stacklevel=4)
    return fallback