def login(self, request, session, creds, segments):
        """
        Called to check the credentials of a user.

        Here we extend guard's implementation to preauthenticate users if they
        have a valid persistent session.

        @type request: L{nevow.inevow.IRequest}
        @param request: The HTTP request being handled.

        @type session: L{nevow.guard.GuardSession}
        @param session: The user's current session.

        @type creds: L{twisted.cred.credentials.ICredentials}
        @param creds: The credentials the user presented.

        @type segments: L{tuple}
        @param segments: The remaining segments of the URL.

        @return: A deferred firing with the user's avatar.
        """
        self._maybeCleanSessions()

        if isinstance(creds, credentials.Anonymous):
            preauth = self.authenticatedUserForKey(session.uid)
            if preauth is not None:
                self.savorSessionCookie(request)
                creds = userbase.Preauthenticated(preauth)

        def cbLoginSuccess(input):
            """
            User authenticated successfully.

            Create the persistent session, and associate it with the
            username. (XXX it doesn't work like this now)
            """
            user = request.args.get('username')
            if user is not None:
                # create a database session and associate it with this user
                cookieValue = session.uid
                if request.args.get('rememberMe'):
                    self.createSessionForKey(cookieValue, creds.username)
                    self.savorSessionCookie(request)
            return input

        return (
            guard.SessionWrapper.login(
                self, request, session, creds, segments)
            .addCallback(cbLoginSuccess))