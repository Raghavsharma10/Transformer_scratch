def set_cell(self, index, column, value):
        """
        Sets the value of a single cell. If the index and/or column is not in the current index/columns then a new
        index and/or column will be created.

        :param index: index value
        :param column: column name
        :param value: value to set
        :return: nothing
        """
        if self._sort:
            exists, i = sorted_exists(self._index, index)
            if not exists:
                self._insert_row(i, index)
        else:
            try:
                i = self._index.index(index)
            except ValueError:
                i = len(self._index)
                self._add_row(index)
        try:
            c = self._columns.index(column)
        except ValueError:
            c = len(self._columns)
            self._add_column(column)
        self._data[c][i] = value