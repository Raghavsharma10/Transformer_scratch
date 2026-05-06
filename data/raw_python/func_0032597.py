def head(self, request, website):
        """
        Provide content to include in the head of the document.  If you only
        need to provide a stylesheet, see L{stylesheetLocation}.  Otherwise,
        override this.

        @type request: L{inevow.IRequest} provider
        @param request: The request object for which this is a response.

        @param website: The site-wide L{xmantissa.website.WebSite} instance.
            Primarily of interest for its C{rootURL} method.

        @return: Anything providing or adaptable to L{nevow.inevow.IRenderer},
            or C{None} to include nothing.
        """
        stylesheet = self.stylesheetLocation
        if stylesheet is not None:
            root = website.rootURL(request)
            for segment in stylesheet:
                root = root.child(segment)
            return tags.link(rel='stylesheet', type='text/css', href=root)