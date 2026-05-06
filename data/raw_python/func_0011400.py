def _pfp__snapshot(self, recurse=True):
        """Save off the current value of the field
        """
        if hasattr(self, "_pfp__value"):
            self._pfp__snapshot_value = self._pfp__value