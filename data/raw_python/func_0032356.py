def resetPassword(self, attempt, newPassword):
        """
        @param attempt: L{_PasswordResetAttempt}

        reset the password of the user who initiated C{attempt} to C{newPassword},
        and afterward, delete the attempt and any persistent sessions that belong
        to the user
        """

        self.accountByAddress(attempt.username).password = newPassword

        self.store.query(
            PersistentSession,
            PersistentSession.authenticatedAs == str(attempt.username)
            ).deleteFromStore()

        attempt.deleteFromStore()