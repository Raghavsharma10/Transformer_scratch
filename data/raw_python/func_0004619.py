def to_dict(self, index=True, ordered=False):
        """
        Returns a dict where the keys are the data and index names and the values are list of the data and index.

        :param index: If True then include the index in the dict with the index_name as the key
        :param ordered: If True then return an OrderedDict() to preserve the order of the columns in the Series
        :return: dict or OrderedDict()
        """
        result = OrderedDict() if ordered else dict()
        if index:
            result.update({self._index_name: self._index})
        if ordered:
            data_dict = [(self._data_name, self._data)]
        else:
            data_dict = {self._data_name: self._data}
        result.update(data_dict)
        return result