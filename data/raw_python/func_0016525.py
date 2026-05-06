def cancel(self):
        """Cancel a running :meth:`iterconsume` session."""
        if self.channel_open:
            try:
                self.backend.cancel(self.consumer_tag)
            except KeyError:
                pass