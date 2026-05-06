def locateChild(self, ctx, segments):
        """
        Initialize self with the given key's L{_PasswordResetAttempt}, if any.

        @param segments: a L{_PasswordResetAttempt} key (hopefully)
        @return: C{(self, ())} with C{self.attempt} initialized, or L{NotFound}
        @see: L{attemptByKey}
        """
        if len(segments) == 1:
            attempt = self.attemptByKey(unicode(segments[0]))
            if attempt is not None:
                self.attempt = attempt
                return (self, ())
        return NotFound