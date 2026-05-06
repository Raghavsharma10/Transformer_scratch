def produceResource(self, request, segments, webViewer):
        """
        Produce a resource that traverses site-wide content, passing down the
        given webViewer.  This delegates to the site store's
        L{IMantissaSite} adapter, to avoid a conflict with the
        L{ISiteRootPlugin} interface.

        This method will typically be given an L{_AuthenticatedWebViewer}, which
        can build an appropriate resource for an authenticated shell page,
        whereas the site store's L{IWebViewer} adapter would show an anonymous
        page.

        The result of this method will be a L{_CustomizingResource}, to provide
        support for resources which may provide L{ICustomizable}.  Note that
        Mantissa itself no longer implements L{ICustomizable} anywhere, though.
        All application code should phase out inspecting the string passed to
        ICustomizable in favor of getting more structured information from the
        L{IWebViewer}.  However, it has not been deprecated yet because
        the interface which allows application code to easily access the
        L{IWebViewer} from view code has not yet been developed; it is
        forthcoming.

        See ticket #2707 for progress on this.
        """
        mantissaSite = self.publicSiteRoot
        if mantissaSite is not None:
            for resource, domain in userbase.getAccountNames(self.store):
                username = '%s@%s' % (resource, domain)
                break
            else:
                username = None
            bottomResource, newSegments = mantissaSite.siteProduceResource(
                request, segments, webViewer)
            return (_CustomizingResource(bottomResource, username), newSegments)
        return None