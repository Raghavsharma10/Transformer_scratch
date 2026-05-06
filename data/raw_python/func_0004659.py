def _add_column(self, column):
        """
        Add a new column to the DataFrame

        :param column: column name
        :return: nothing
        """
        self._columns.append(column)
        if self._blist:
            self._data.append(blist([None] * len(self._index)))
        else:
            self._data.append([None] * len(self._index))