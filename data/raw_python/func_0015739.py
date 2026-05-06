def _from_gerror(cls, error, own=True):
        """Creates a GError exception and takes ownership if own is True"""

        if not own:
            error = error.copy()

        self = cls()
        self._error = error
        return self