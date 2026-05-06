def getDocFactory(self, name, default=None):
        """
        Locate a L{nevow.inevow.IDocFactory} object with the given name from
        the themes installed on the site store and return it.
        """
        loader = None
        for theme in getInstalledThemes(self.siteStore):
            loader = theme.getDocFactory(name)
            if loader is not None:
                return loader
        return default