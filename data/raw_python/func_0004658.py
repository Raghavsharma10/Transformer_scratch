def _add_row(self, index):
        """
        Add a new row to the DataFrame

        :param index: index of the new row
        :return: nothing
        """
        self._index.append(index)
        for c, _ in enumerate(self._columns):
            self._data[c].append(None)