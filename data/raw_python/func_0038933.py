def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """
        if isinstance(col, pd.Series):
            col = col.to_frame()

        output = pd.DataFrame(index=col.index)
        output[self.col_name] = col.apply(self.safe_date, axis=1)

        return output