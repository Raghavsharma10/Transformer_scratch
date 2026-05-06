def _set_categories(self):
        """
        Inplace conversion from categories.
        """
        for column, _ in self._categories.items():
            if column in self.columns:
                self[column] = self[column].astype('category')