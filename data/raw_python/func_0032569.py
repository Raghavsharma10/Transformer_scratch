def getKeyForAPI(cls, siteStore, apiName):
        """
        Get the API key for the named API, if one exists.

        @param siteStore: The site store.
        @type siteStore: L{axiom.store.Store}

        @param apiName: The name of the API.
        @type apiName: C{unicode} (L{APIKey} constant)

        @rtype: L{APIKey} or C{NoneType}
        """
        return siteStore.findUnique(
            cls, cls.apiName == apiName, default=None)