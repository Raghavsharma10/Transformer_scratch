def cleartextRoot(self, hostname=None):
        """
        Return a string representing the HTTP URL which is at the root of this
        site.

        @param hostname: An optional unicode string which, if specified, will
        be used as the hostname in the resulting URL, regardless of the
        C{hostname} attribute of this item.
        """
        warnings.warn(
            "Use ISiteURLGenerator.rootURL instead of WebSite.cleartextRoot.",
            category=DeprecationWarning,
            stacklevel=2)
        if self.store.parent is not None:
            generator = ISiteURLGenerator(self.store.parent)
        else:
            generator = ISiteURLGenerator(self.store)
        return generator.cleartextRoot(hostname)