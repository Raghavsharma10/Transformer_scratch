def beforeRender(self, ctx):
        """
        Before rendering, retrieve the hostname from the request being
        responded to and generate an URL which will serve as the root for
        all JavaScript modules to be loaded.
        """
        request = IRequest(ctx)
        root = self.webSite.rootURL(request)
        self._moduleRoot = root.child('__jsmodule__')