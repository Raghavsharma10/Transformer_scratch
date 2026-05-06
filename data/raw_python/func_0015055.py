def startThread(self):
        """Spawns new NSThread to handle notifications."""
        if self._thread is not None:
            return
        self._thread = NSThread.alloc().initWithTarget_selector_object_(self, 'runPowerNotificationsThread', None)
        self._thread.start()