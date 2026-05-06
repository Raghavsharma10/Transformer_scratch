def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """
        out = pd.DataFrame(index=col.index)
        out[self.col_name] = col.fillna(self.default_value)
        out[self.new_name] = (pd.notnull(col) * 1).astype(int)
        return out