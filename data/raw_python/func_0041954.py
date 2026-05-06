def commit_all(self):
        """Commits all current nesting transactions."""
        while self._transaction_nesting_level != 0:
            if not self._auto_commit and self._transaction_nesting_level == 1:
                return self.commit()
            self.commit()