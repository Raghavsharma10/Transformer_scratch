def _maybeCleanSessions(self):
        """
        Clean expired sessions if it's been long enough since the last clean.
        """
        sinceLast = self._clock.seconds() - self._lastClean
        if sinceLast > self.sessionCleanFrequency:
            self._cleanSessions()