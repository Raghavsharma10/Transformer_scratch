def installOffering(self, offering, configuration):
        """
        Create an app store for an L{Offering} and install its
        dependencies. Also create an L{InstalledOffering} in the site store,
        and return it.
        """
        s = self.store.parent
        self.installedOfferingCount += 1
        return installOffering(s, offering, configuration)