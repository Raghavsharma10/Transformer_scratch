def itertuples(self, index=True, name='Raccoon'):
        """
        Iterates over DataFrame rows as tuple of the values.

        :param index: if True then include the index
        :param name: name of the namedtuple
        :return: namedtuple
        """
        fields = [self._index_name] if index else list()
        fields.extend(self._columns)
        row_tuple = namedtuple(name, fields)
        for i in range(len(self._index)):
            row = {self._index_name: self._index[i]} if index else dict()
            for c, col in enumerate(self._columns):
                row[col] = self._data[c][i]
            yield row_tuple(**row)