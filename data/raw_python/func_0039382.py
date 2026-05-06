def transform(self, column):
        """Applies an exponential to values to turn them positive numbers.

        Args:
            column (pandas.DataFrame): Data to transform.

        Returns:
            pd.DataFrame
        """
        self.check_data_type()

        return pd.DataFrame({self.col_name: np.exp(column[self.col_name])})