def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """
        output = pd.DataFrame()
        new_name = '?' + self.col_name

        col.loc[col[new_name] == 0, self.col_name] = np.nan
        output[self.col_name] = col[self.col_name]
        return output