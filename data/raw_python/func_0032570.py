def setKeyForAPI(cls, siteStore, apiName, apiKey):
        """
        Set the API key for the named API, overwriting any existing key.

        @param siteStore: The site store to install the key in.
        @type siteStore: L{axiom.store.Store}

        @param apiName: The name of the API.
        @type apiName: C{unicode} (L{APIKey} constant)

        @param apiKey: The key for accessing the API.
        @type apiKey: C{unicode}

        @rtype: L{APIKey}
        """
        existingKey = cls.getKeyForAPI(siteStore, apiName)
        if existingKey is None:
            return cls(store=siteStore, apiName=apiName, apiKey=apiKey)
        existingKey.apiKey = apiKey
        return existingKey