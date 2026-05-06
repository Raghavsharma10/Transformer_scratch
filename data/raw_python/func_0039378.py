def fit_transform(self, col):
        """Prepare the transformer and return processed data.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        if self.anonymize:
            col = self.anonymize_column(col)

        self._fit(col)
        return self.transform(col)