def getModule(self, moduleName):
        """
        Retrieve a JavaScript module cache from the file path cache.

        @returns: Module cache for the named module.
        @rtype: L{CachedJSModule}
        """
        if moduleName not in self.moduleCache:
            modulePath = FilePath(
                athena.jsDeps.getModuleForName(moduleName)._cache.path)
            cachedModule = self.moduleCache[moduleName] = CachedJSModule(
                moduleName, modulePath)
        else:
            cachedModule = self.moduleCache[moduleName]
        return cachedModule