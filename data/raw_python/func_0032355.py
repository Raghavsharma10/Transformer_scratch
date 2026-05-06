def newAttemptForUser(self, user):
        """
        Create an L{_PasswordResetAttempt} for the user whose username is C{user}
        @param user: C{unicode} username
        """
        # we could query for other attempts by the same
        # user within some timeframe and raise an exception,
        # if we wanted
        return _PasswordResetAttempt(store=self.store,
                                     username=user,
                                     timestamp=extime.Time(),
                                     key=self._makeKey(user))