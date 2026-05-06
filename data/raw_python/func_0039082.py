def transform(self, col):
        """Prepare the transformer to convert data and return the processed table.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        out = pd.DataFrame(index=col.index)
        out[self.col_name] = col.apply(self.get_val, axis=1)

        if self.subtype == 'int':
            out[self.col_name] = out[self.col_name].astype(int)

        return out