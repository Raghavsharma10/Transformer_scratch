def set_column(self, index=None, column=None, values=None):
        """
        Set a column to a single value or list of values. If any of the index values are not in the current indexes
        then a new row will be created.

        :param index: list of index values or list of booleans. If a list of booleans then the list must be the same\
        length as the DataFrame
        :param column: column name
        :param values: either a single value or a list. The list must be the same length as the index list if the index\
        list is values, or the length of the True values in the index list if the index list is booleans
        :return: nothing
        """
        try:
            c = self._columns.index(column)
        except ValueError:  # new column
            c = len(self._columns)
            self._add_column(column)
        if index:  # index was provided
            if all([isinstance(i, bool) for i in index]):  # boolean list
                if not isinstance(values, (list, blist)):  # single value provided, not a list, so turn values into list
                    values = [values for x in index if x]
                if len(index) != len(self._index):
                    raise ValueError('boolean index list must be same size of existing index')
                if len(values) != index.count(True):
                    raise ValueError('length of values list must equal number of True entries in index list')
                indexes = [i for i, x in enumerate(index) if x]
                for x, i in enumerate(indexes):
                    self._data[c][i] = values[x]
            else:  # list of index
                if not isinstance(values, (list, blist)):  # single value provided, not a list, so turn values into list
                    values = [values for _ in index]
                if len(values) != len(index):
                    raise ValueError('length of values and index must be the same.')
                # insert or append indexes as needed
                if self._sort:
                    exists_tuples = list(zip(*[sorted_exists(self._index, x) for x in index]))
                    exists = exists_tuples[0]
                    indexes = exists_tuples[1]
                    if not all(exists):
                        self._insert_missing_rows(index)
                        indexes = [sorted_index(self._index, x) for x in index]
                else:
                    try:  # all index in current index
                        indexes = [self._index.index(x) for x in index]
                    except ValueError:  # new rows need to be added
                        self._add_missing_rows(index)
                        indexes = [self._index.index(x) for x in index]
                for x, i in enumerate(indexes):
                    self._data[c][i] = values[x]
        else:  # no index, only values
            if not isinstance(values, (list, blist)):  # values not a list, turn into one of length same as index
                values = [values for _ in self._index]
            if len(values) != len(self._index):
                raise ValueError('values list must be at same length as current index length.')
            else:
                self._data[c] = blist(values) if self._blist else values