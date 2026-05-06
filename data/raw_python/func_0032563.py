def renderHTTP(self, context):
        """
        Render C{self.resource} through a L{StylesheetRewritingRequestWrapper}.
        """
        request = IRequest(context)
        request = StylesheetRewritingRequestWrapper(
            request, self.installedOfferingNames, self.rootURL)
        context.remember(request, IRequest)
        return self.resource.renderHTTP(context)