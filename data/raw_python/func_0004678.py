def iterrows(self, index=True):
        """
        Iterates over DataFrame rows as dictionary of the values. The index will be included.

        :param index: if True include the index in the results
        :return: dictionary
        """
        for i in range(len(self._index)):
            row = {self._index_name: self._index[i]} if index else dict()
            for c, col in enumerate(self._columns):
                row[col] = self._data[c][i]
            yield row