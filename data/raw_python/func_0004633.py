def append_row(self, index, value):
        """
        Appends a row of value to the end of the data. Be very careful with this function as for sorted Series it will 
        not enforce sort order. Use this only for speed when needed, be careful.

        :param index: index
        :param value: value
        :return: nothing
        """
        if index in self._index:
            raise IndexError('index already in Series')

        self._index.append(index)
        self._data.append(value)