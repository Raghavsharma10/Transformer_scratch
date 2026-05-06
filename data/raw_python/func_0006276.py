def stop_event_stream(self):
        """ Stop streaming events from `gerrit stream-events`."""
        if self._stream:
            self._stream.stop()
            self._stream.join()
            self._stream = None
            with self._events.mutex:
                self._events.queue.clear()