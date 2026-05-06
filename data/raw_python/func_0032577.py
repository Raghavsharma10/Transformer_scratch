def renderHTTP(self, context):
        """
        Check to see if the wrapped resource wants to be rendered over HTTPS
        and generate a redirect if this is so, if HTTPS is available, and if
        the request is not already over HTTPS.
        """
        if getattr(self.wrappedResource, 'needsSecure', False):
            request = IRequest(context)
            url = self.urlGenerator.encryptedRoot()
            if url is not None:
                for seg in request.prepath:
                    url = url.child(seg)
                return url
        return self.wrappedResource.renderHTTP(context)