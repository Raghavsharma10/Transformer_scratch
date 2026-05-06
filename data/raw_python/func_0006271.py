def stop(self):
        """ Stop the thread. """
        self._stop.set()
        if self._channel is not None:
            self._channel.close()