def removeSessionWithKey(self, key):
        """
        Remove a persistent session, if it exists.

        @type key: L{bytes}
        @param key: The persistent session identifier.
        """
        self.store.query(
            PersistentSession,
            PersistentSession.sessionKey == key).deleteFromStore()