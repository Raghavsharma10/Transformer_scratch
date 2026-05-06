def to_timestamp(self, data):
        """Transform a datetime series into linux epoch.

        Args:
            data(pandas.DataFrame): DataFrame containins a column named as `self.col_name`.

        Returns:
            pandas.Series
        """
        result = pd.Series(index=data.index)
        _slice = ~data[self.col_name].isnull()

        result[_slice] = data[_slice][self.col_name].astype('int64')
        return result