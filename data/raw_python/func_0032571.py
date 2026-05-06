def rootURL(self, request):
        """
        Return the URL for the root of this website which is appropriate to use
        in links generated in response to the given request.

        @type request: L{twisted.web.http.Request}
        @param request: The request which is being responded to.

        @rtype: L{URL}
        @return: The location at which the root of the resource hierarchy for
            this website is available.
        """
        host = request.getHeader('host') or self.hostname
        if ':' in host:
            host = host.split(':', 1)[0]
        for domain in [self.hostname] + getDomainNames(self.store):
            if (host == domain or
                host.startswith('www.') and host[len('www.'):] == domain):
                return URL(scheme='', netloc='', pathsegs=[''])
        if request.isSecure():
            return self.encryptedRoot(self.hostname)
        else:
            return self.cleartextRoot(self.hostname)