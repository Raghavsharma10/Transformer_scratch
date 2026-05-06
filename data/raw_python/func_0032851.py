def indirect(self, interface):
        """
        Create a L{VirtualHostWrapper} so it can have the first chance to
        handle web requests.
        """
        if interface is IResource:
            siteStore = self.store.parent
            if self.store.parent is None:
                siteStore = self.store
            return VirtualHostWrapper(
                siteStore,
                IWebViewer(self.store),
                self)
        return self