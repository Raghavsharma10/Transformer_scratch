def getHeadContent(self, req):
        """
        Retrieve a list of header content from all installed themes on the site
        store.
        """
        site = ixmantissa.ISiteURLGenerator(self.store)
        for t in getInstalledThemes(self.store):
            yield t.head(req, site)