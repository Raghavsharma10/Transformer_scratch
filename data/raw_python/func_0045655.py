def get_kwargs(self):
        """Return kwargs from attached attributes."""
        return {k: v for k, v in vars(self).items() if k not in self._ignored}