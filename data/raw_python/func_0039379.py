def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        output = pd.DataFrame()
        output[self.col_name] = self.get_category(col[self.col_name])

        return output