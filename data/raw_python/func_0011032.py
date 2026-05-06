def send(self, timeout=None):
        """Returns an event or None if no events occur before timeout."""
        if self.sigint_event and is_main_thread():
            with ReplacedSigIntHandler(self.sigint_handler):
                return self._send(timeout)
        else:
            return self._send(timeout)