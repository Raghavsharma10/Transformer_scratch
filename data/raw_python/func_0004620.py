def head(self, rows):
        """
        Return a Series of the first N rows

        :param rows: number of rows
        :return: Series
        """
        rows_bool = [True] * min(rows, len(self._index))
        rows_bool.extend([False] * max(0, len(self._index) - rows))
        return self.get(indexes=rows_bool)