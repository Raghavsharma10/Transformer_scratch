def createResource(self):
        """
        When invoked by L{PrefixURLMixin}, return a L{websharing.SharingIndex}
        for my application.
        """
        pp = ixmantissa.IPublicPage(self.application, None)
        if pp is not None:
            warn(
            "Use the sharing system to provide public pages, not IPublicPage",
            category=DeprecationWarning,
            stacklevel=2)
            return pp.getResource()
        return SharingIndex(self.application.open())