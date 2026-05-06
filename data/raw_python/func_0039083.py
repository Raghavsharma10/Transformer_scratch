def reverse_transform(self, col):
        """Converts data back into original format.

        Args:
            col(pandas.DataFrame): Data to transform.

        Returns:
            pandas.DataFrame
        """

        output = pd.DataFrame(index=col.index)
        output[self.col_name] = col.apply(self.safe_round, axis=1)

        if self.subtype == 'int':
            output[self.col_name] = output[self.col_name].astype(int)

        return output