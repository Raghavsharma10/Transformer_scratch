def cookieDomainForRequest(self, request):
        """
        Pick a domain to use when setting cookies.

        @type request: L{nevow.inevow.IRequest}
        @param request: Request to determine cookie domain for

        @rtype: C{str} or C{None}
        @return: Domain name to use when setting cookies, or C{None} to
            indicate that only the domain in the request should be used
        """
        host = request.getHeader('host')
        if host is None:
            # This is a malformed request that we cannot possibly handle
            # safely, fall back to the default behaviour.
            return None

        host = host.split(':')[0]
        for domain in self._domains:
            suffix = "." + domain
            if host == domain:
                # The request is for a domain which is directly recognized.
                if self._enableSubdomains:
                    # Subdomains are enabled, so the suffix is returned to
                    # enable the cookie for this domain and all its subdomains.
                    return suffix

                # Subdomains are not enabled, so None is returned to allow the
                # default restriction, which will enable this cookie only for
                # the domain in the request, to apply.
                return None

            if self._enableSubdomains and host.endswith(suffix):
                # The request is for a subdomain of a directly recognized
                # domain and subdomains are enabled.  Drop the unrecognized
                # subdomain portion and return the suffix to enable the cookie
                # for this domain and all its subdomains.
                return suffix

        if self._enableSubdomains:
            # No directly recognized domain matched the request.  If subdomains
            # are enabled, prefix the request domain with "." to make the
            # cookie valid for that domain and all its subdomains.  This
            # probably isn't extremely useful.  Perhaps it shouldn't work this
            # way.
            return "." + host

        # Subdomains are disabled and the domain from the request was not
        # recognized.  Return None to get the default behavior.
        return None