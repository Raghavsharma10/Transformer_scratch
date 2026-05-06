def get_rows(self, indexes, as_list=False):
        """
        For a list of indexes return the values of the indexes in that column.

        :param indexes: either a list of index values or a list of booleans with same length as all indexes
        :param as_list: if True return a list, if False return Series
        :return: Series if as_list if False, a list if as_list is True
        """
        if all([isinstance(i, bool) for i in indexes]):  # boolean list
            if len(indexes) != len(self._index):
                raise ValueError('boolean index list must be same size of existing index')
            if all(indexes):  # the entire column
                data = self._data
                index = self._index
            else:
                data = list(compress(self._data, indexes))
                index = list(compress(self._index, indexes))
        else:  # index values list
            locations = [sorted_index(self._index, x) for x in indexes] if self._sort \
                else [self._index.index(x) for x in indexes]
            data = [self._data[i] for i in locations]
            index = [self._index[i] for i in locations]
        return data if as_list else Series(data=data, index=index, data_name=self._data_name,
                                           index_name=self._index_name, sort=self._sort)