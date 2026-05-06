def _on_closed(self):
        """Invoked by connections when they are closed."""
        self._connected.clear()
        if not self._closing:
            if self._on_close_callback:
                self._on_close_callback()
            else:
                raise exceptions.ConnectionError('closed')