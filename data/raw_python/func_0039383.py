def reverse_transform(self, column):
        """Applies the natural logarithm function to turn positive values into real ranged values.

        Args:
            column (pandas.DataFrame): Data to transform.

        Returns:
            pd.DataFrame
        """
        self.check_data_type()

        return pd.DataFrame({self.col_name: np.log(column[self.col_name])})