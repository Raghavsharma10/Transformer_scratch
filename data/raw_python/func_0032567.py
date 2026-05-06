def rootURL(self, request):
        """
        Simple utility function to provide a root URL for this website which is
        appropriate to use in links generated in response to the given request.

        @type request: L{twisted.web.http.Request}
        @param request: The request which is being responded to.

        @rtype: L{URL}
        @return: The location at which the root of the resource hierarchy for
            this website is available.
        """
        warnings.warn(
            "Use ISiteURLGenerator.rootURL instead of WebSite.rootURL.",
            category=DeprecationWarning,
            stacklevel=2)
        if self.store.parent is not None:
            generator = ISiteURLGenerator(self.store.parent)
        else:
            generator = ISiteURLGenerator(self.store)
        return generator.rootURL(request)