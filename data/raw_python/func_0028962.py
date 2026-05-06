def _reset_changes(self):
        """Stores current values for comparison later"""
        self._original = {}
        if self.last_updated is not None:
            self._original['last_updated'] = self.last_updated