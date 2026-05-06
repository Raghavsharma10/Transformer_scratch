def join(self, timeout=None):
        """Wait for this Task to end. If a timeout is given, after the time expires the function
        will return anyway."""
        if not self._started:
            raise RuntimeError('cannot join task before it is started')
        return self._exit_event.wait(timeout)