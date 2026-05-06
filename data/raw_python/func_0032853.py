def locateChild(self, context, segments):
        """
        Delegate dispatch to a sharing resource if the request is for a user
        subdomain, otherwise fall back to the wrapped resource's C{locateChild}
        implementation.
        """
        request = IRequest(context)
        hostname = request.getHeader('host')

        info = self.subdomain(hostname)
        if info is not None:
            username, domain = info
            index = UserIndexPage(IRealm(self.siteStore),
                                  self.webViewer)
            resource = index.locateChild(None, [username])[0]
            return resource, segments
        return self.wrapped.locateChild(context, segments)