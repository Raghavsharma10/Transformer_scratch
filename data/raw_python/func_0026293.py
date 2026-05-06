def skipline(self):
        """
        Skip the next line and returns position and size of line.
        Raises IOError if pre- and suffix of line do not match.
        """
        position = self.tell()
        prefix = self._fix()
        self.seek(prefix, 1)  # skip content
        suffix = self._fix()

        if prefix != suffix:
            raise IOError(_FIX_ERROR)

        return position, prefix