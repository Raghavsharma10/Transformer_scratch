def explicitLogout(self, session):
        """
        Handle a user-requested logout.

        Here we override guard's behaviour for the logout action to delete the
        persistent session.  In this case the user has explicitly requested a
        logout, so the persistent session must be deleted to require the user
        to log in on the next request.

        @type session: L{nevow.guard.GuardSession}
        @param session: The session of the user logging out.
        """
        guard.SessionWrapper.explicitLogout(self, session)
        self.removeSessionWithKey(session.uid)