def _error(self, message, start, end=None):
        """Raise a nice error, with the token highlighted."""
        raise errors.EfilterParseError(
            source=self.source, start=start, end=end, message=message)