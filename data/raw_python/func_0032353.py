def attemptByKey(self, key):
        """
        Locate the L{_PasswordResetAttempt} that corresponds to C{key}
        """

        return self.store.findUnique(_PasswordResetAttempt,
                                     _PasswordResetAttempt.key == key,
                                     default=None)