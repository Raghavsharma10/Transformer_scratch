def get_entire_column(self, column, as_list=False):
        """
        Shortcut method to retrieve a single column all rows. Since this is a common use case this method will be
        faster than the more general method.

        :param column: single column name
        :param as_list: if True return a list, if False return DataFrame
        :return: DataFrame is as_list if False, a list if as_list is True
        """
        c = self._columns.index(column)
        data = self._data[c]
        return data if as_list else DataFrame(data={column: data}, index=self._index, index_name=self._index_name,
                                              sort=self._sort)