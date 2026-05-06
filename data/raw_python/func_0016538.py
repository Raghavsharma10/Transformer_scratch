def cancel(self):
        """Cancel a running :meth:`iterconsume` session."""
        for consumer_tag in self._open_consumers.values():
            try:
                self.backend.cancel(consumer_tag)
            except KeyError:
                pass
        self._open_consumers.clear()