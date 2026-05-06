def _preferredThemes(self):
        """
        Return a list of themes in the order of preference that this user has
        selected via L{PrivateApplication.preferredTheme}.
        """
        themes = getInstalledThemes(self.store.parent)
        _reorderForPreference(themes, self.preferredTheme)
        return themes