def getDocFactory(self, fragmentName, default=None):
        """
        For a given fragment, return a loaded Nevow template.

        @param fragmentName: the name of the template (can include relative
        paths).

        @param default: a default loader; only used if provided and the
        given fragment name cannot be resolved.

        @return: A loaded Nevow template.
        @type return: L{nevow.loaders.xmlfile}
        """
        if fragmentName in self.cachedLoaders:
            return self.cachedLoaders[fragmentName]
        segments = fragmentName.split('/')
        segments[-1] += '.html'
        file = self.directory
        for segment in segments:
            file = file.child(segment)
        if file.exists():
            loader = xmlfile(file.path)
            self.cachedLoaders[fragmentName] = loader
            return loader
        return default