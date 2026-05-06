def locateChild(self, context, segments):
        """
        Find the offering with the name matching the first segment and return a
        L{File} for its I{staticContentPath}.
        """
        name = segments[0]
        try:
            staticContent = self.staticPaths[name]
        except KeyError:
            return NotFound
        else:
            resource = File(staticContent.path)
            resource.processors = self.processors
            return resource, segments[1:]
        return NotFound