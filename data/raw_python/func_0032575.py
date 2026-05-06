def locateChild(self, context, segments):
        """
        Unwrap the wrapped resource if HTTPS is already being used, otherwise
        wrap it in a helper which will preserve the wrapping all the way down
        to the final resource.
        """
        request = IRequest(context)
        if request.isSecure():
            return self.wrappedResource, segments
        return _SecureWrapper(self.urlGenerator, self.wrappedResource), segments