def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """
        out = pd.DataFrame()
        out[self.col_name] = self.safe_datetime_cast(col)
        out[self.col_name] = self.to_timestamp(out)

        return out