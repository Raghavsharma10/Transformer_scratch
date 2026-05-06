def produceResource(self, request, segments, webViewer):
        """
        Return a C{(resource, subsegments)} tuple or None, depending on whether
        I wish to return an L{IResource} provider for the given set of segments
        or not.
        """
        def thunk():
            cr = getattr(self, 'createResource', None)
            if cr is not None:
                return cr()
            else:
                return self.createResourceWith(webViewer)
        return self._produceIt(segments, thunk)