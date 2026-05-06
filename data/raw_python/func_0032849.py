def locateChild(self, context, segments):
        """
        Return a statically defined child or a child defined by a site root
        plugin or an avatar from guard.
        """
        request = IRequest(context)
        webViewer = IWebViewer(self.store, None)
        childAndSegments = self.siteProduceResource(request, segments, webViewer)
        if childAndSegments is not None:
            return childAndSegments
        return NotFound