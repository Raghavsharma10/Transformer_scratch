def _revert_categories(self):
        """
        Inplace conversion to categories.
        """
        for column, dtype in self._categories.items():
            if column in self.columns:
                self[column] = self[column].astype(dtype)