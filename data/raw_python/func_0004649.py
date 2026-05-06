def get_cell(self, index, column):
        """
        For a single index and column value return the value of the cell

        :param index: index value
        :param column: column name
        :return: value
        """
        i = sorted_index(self._index, index) if self._sort else self._index.index(index)
        c = self._columns.index(column)
        return self._data[c][i]