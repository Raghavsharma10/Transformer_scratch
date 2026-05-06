def _add_row(self, index):
        """
        Add a new row to the Series

        :param index: index of the new row
        :return: nothing
        """
        self._index.append(index)
        self._data.append(None)