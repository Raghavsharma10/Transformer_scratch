def feed(self, data):
        """
        Feed new data into this pipe.  This method is assumed to be called
        from a separate thread, so synchronization is done.
        
        @param data: the data to add
        @type data: str
        """
        self._lock.acquire()
        try:
            if self._event is not None:
                self._event.set()
            self._buffer.fromstring(data)
            self._cv.notifyAll()
        finally:
            self._lock.release()