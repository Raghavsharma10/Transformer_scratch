def renderHTTP(self, context):
        """
        Render the wrapped resource if HTTPS is already being used, otherwise
        invoke a helper which may generate a redirect.
        """
        request = IRequest(context)
        if request.isSecure():
            renderer = self.wrappedResource
        else:
            renderer = _SecureWrapper(self.urlGenerator, self.wrappedResource)
        return renderer.renderHTTP(context)