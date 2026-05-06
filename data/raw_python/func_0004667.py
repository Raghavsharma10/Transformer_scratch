def to_dict(self, index=True, ordered=False):
        """
        Returns a dict where the keys are the column names and the values are lists of the values for that column.

        :param index: If True then include the index in the dict with the index_name as the key
        :param ordered: If True then return an OrderedDict() to preserve the order of the columns in the DataFrame
        :return: dict or OrderedDict()
        """
        result = OrderedDict() if ordered else dict()
        if index:
            result.update({self._index_name: self._index})
        if ordered:
            data_dict = [(column, self._data[i]) for i, column in enumerate(self._columns)]
        else:
            data_dict = {column: self._data[i] for i, column in enumerate(self._columns)}
        result.update(data_dict)
        return result