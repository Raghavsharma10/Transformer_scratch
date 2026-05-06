def set_cell(self, index, value):
        """
        Sets the value of a single cell. If the index is not in the current index then a new index will be created.

        :param index: index value
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
        self._data[i] = value