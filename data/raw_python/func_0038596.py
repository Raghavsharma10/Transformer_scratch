def deadline(self):
        """Return next day as deadline if no deadline provided."""
        if not self._deadline:
            self._deadline = self.now + timezone.timedelta(days=1)
        return self._deadline