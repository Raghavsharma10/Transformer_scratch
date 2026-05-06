def getAllThemes(self):
        """
        Collect themes from all available offerings, or (if called
        multiple times) return the previously collected list.
        """
        if self._getAllThemesCache is None:
            self._getAllThemesCache = self._realGetAllThemes()
        return self._getAllThemesCache