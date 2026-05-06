def stopThread(self):
        """Stops spawned NSThread."""
        if self._thread is not None:
            self.performSelector_onThread_withObject_waitUntilDone_('stopPowerNotificationsThread', self._thread, None, objc.YES)
            self._thread = None