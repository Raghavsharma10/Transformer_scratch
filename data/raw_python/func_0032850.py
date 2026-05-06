def siteProduceResource(self, req, segments, webViewer):
        """
        Retrieve a child resource and segments from rootChild_ methods on this
        object and SiteRootPlugins.

        @return: a 2-tuple of (resource, segments), suitable for return from
        locateChild.

        @param req: an L{IRequest} provider.

        @param segments: a tuple of L{str}s, the segments from the request.

        @param webViewer: an L{IWebViewer}, to be propagated through the child
        lookup process.
        """

        # rootChild_* is not the same as child_, because its signature is
        # different.  Maybe this should be done some other way.
        shortcut = getattr(self, 'rootChild_' + segments[0], None)
        if shortcut:
            res = shortcut(req, webViewer)
            if res is not None:
                return res, segments[1:]

        for plg in self.store.powerupsFor(ISiteRootPlugin):
            produceResource = getattr(plg, 'produceResource', None)
            if produceResource is not None:
                childAndSegments = produceResource(req, segments, webViewer)
            else:
                childAndSegments = plg.resourceFactory(segments)
            if childAndSegments is not None:
                return childAndSegments
        return None