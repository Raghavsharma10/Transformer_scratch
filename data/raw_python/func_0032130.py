def locateChild(self, ctx, segments):
        """
        Retrieve an L{inevow.IResource} to render the contents of the given
        module.
        """
        if len(segments) != 2:
            return NotFound
        hashCode, moduleName = segments
        cachedModule = self.getModule(moduleName)
        return static.Data(
            cachedModule.fileContents,
            'text/javascript', expires=(60 * 60 * 24 * 365 * 5)), []