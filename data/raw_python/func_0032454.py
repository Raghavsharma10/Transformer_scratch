def savorSessionCookie(self, request):
        """
        Make the session cookie last as long as the persistent session.

        @type request: L{nevow.inevow.IRequest}
        @param request: The HTTP request object for the guard login URL.
        """
        cookieValue = request.getSession().uid
        request.addCookie(
            self.cookieKey, cookieValue, path='/',
            max_age=PERSISTENT_SESSION_LIFETIME,
            domain=self.cookieDomainForRequest(request))