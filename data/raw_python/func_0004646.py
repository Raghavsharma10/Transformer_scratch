def _sort_columns(self, columns_list):
        """
        Given a list of column names will sort the DataFrame columns to match the given order

        :param columns_list: list of column names. Must include all column names
        :return: nothing
        """
        if not (all([x in columns_list for x in self._columns]) and all([x in self._columns for x in columns_list])):
            raise ValueError(
                'columns_list must be all in current columns, and all current columns must be in columns_list')
        new_sort = [self._columns.index(x) for x in columns_list]
        self._data = blist([self._data[x] for x in new_sort]) if self._blist else [self._data[x] for x in new_sort]
        self._columns = blist([self._columns[x] for x in new_sort]) if self._blist \
            else [self._columns[x] for x in new_sort]