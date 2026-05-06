def installSite(self):
        """
        Not using the dependency system for this class because it's only
        installed via the command line, and multiple instances can be
        installed.
        """
        for iface, priority in self.__getPowerupInterfaces__([]):
            self.store.powerUp(self, iface, priority)