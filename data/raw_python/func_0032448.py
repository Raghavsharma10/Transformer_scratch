def createSessionForKey(self, key, user):
        """
        Create a persistent session in the database.

        @type key: L{bytes}
        @param key: The persistent session identifier.

        @type user: L{bytes}
        @param user: The username the session will belong to.
        """
        PersistentSession(
            store=self.store,
            sessionKey=key,
            authenticatedAs=user)