def get_escape(self, c, i):
        """Get an escape."""

        try:
            escaped = next(i)
        except StopIteration:
            escaped = ''
        return c + escaped if self.keep_escapes else escaped