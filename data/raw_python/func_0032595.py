def getInstalledThemes(self, store):
        """
        Collect themes from all offerings installed on this store, or (if called
        multiple times) return the previously collected list.
        """
        if not store in self._getInstalledThemesCache:
            self._getInstalledThemesCache[store] = (self.
                                                 _realGetInstalledThemes(store))
        return self._getInstalledThemesCache[store]