def getCredentials(self, request):
        """
        Derive credentials from an HTTP request.

        Override SessionWrapper.getCredentials to add the Host: header to the
        credentials.  This will make web-based virtual hosting work.

        @type request: L{nevow.inevow.IRequest}
        @param request: The request being handled.

        @rtype: L{twisted.cred.credentials.1ICredentials}
        @return: Credentials derived from the HTTP request.
        """
        username = usernameFromRequest(request)
        password = request.args.get('password', [''])[0]
        return credentials.UsernamePassword(username, password)