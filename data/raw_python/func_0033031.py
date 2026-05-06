def getInstalledOfferings(self):
        """
        Return a mapping from the name of each L{InstalledOffering} in
        C{self._siteStore} to the corresponding L{IOffering} plugins.
        """
        d = {}
        installed = self._siteStore.query(InstalledOffering)
        for installation in installed:
            offering = installation.getOffering()
            if offering is not None:
                d[offering.name] = offering
        return d