def stopPowerNotificationsThread(self):
        """Removes the only source from NSRunLoop and cancels thread."""
        assert NSThread.currentThread() == self._thread

        CFRunLoopSourceInvalidate(self._source)
        self._source = None
        NSThread.currentThread().cancel()