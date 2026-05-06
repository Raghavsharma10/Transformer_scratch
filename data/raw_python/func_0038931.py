def fit(self, col):
        """Prepare the transformer to convert data.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            None
        """
        dates = self.safe_datetime_cast(col)
        self.default_val = dates.groupby(dates).count().index[0].timestamp() * 1e9