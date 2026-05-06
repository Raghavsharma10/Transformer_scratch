def _cleanSessions(self):
        """
        Clean expired sesisons.
        """
        tooOld = extime.Time() - timedelta(seconds=PERSISTENT_SESSION_LIFETIME)
        self.store.query(
            PersistentSession,
            PersistentSession.lastUsed < tooOld).deleteFromStore()
        self._lastClean = self._clock.seconds()