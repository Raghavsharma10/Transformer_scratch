def message(self, to, subject, text):
        """Alias for :meth:`compose`."""
        return self.compose(to, subject, text)