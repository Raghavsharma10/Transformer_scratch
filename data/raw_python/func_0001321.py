def overwrite_line(self, n, text):
        """Move back N lines and overwrite line with `text`.
        """
        with self._moveback(n):
            self.term.stream.write(text)