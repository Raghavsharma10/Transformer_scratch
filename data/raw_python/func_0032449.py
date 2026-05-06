def authenticatedUserForKey(self, key):
        """
        Find a persistent session for a user.

        @type key: L{bytes}
        @param key: The persistent session identifier.

        @rtype: L{bytes} or C{None}
        @return: The avatar ID the session belongs to, or C{None} if no such
            session exists.
        """
        session = self.store.findFirst(
            PersistentSession, PersistentSession.sessionKey == key)
        if session is None:
            return None
        else:
            session.renew()
            return session.authenticatedAs