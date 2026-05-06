def get_cell(self, index):
        """
        For a single index and return the value

        :param index: index value
        :return: value
        """
        i = sorted_index(self._index, index) if self._sort else self._index.index(index)
        return self._data[i]