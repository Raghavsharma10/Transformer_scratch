def makeStylesheetResource(self, path, registry):
        """
        Return a resource for the css at the given path with its urls rewritten
        based on self.rootURL.
        """
        return StylesheetRewritingResourceWrapper(
            File(path), self.installedOfferingNames, self.rootURL)