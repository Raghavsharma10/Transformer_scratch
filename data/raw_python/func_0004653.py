def get_matrix(self, indexes, columns):
        """
        For a list of indexes and list of columns return a DataFrame of the values.

        :param indexes: either a list of index values or a list of booleans with same length as all indexes
        :param columns: list of column names
        :return: DataFrame
        """
        if all([isinstance(i, bool) for i in indexes]):  # boolean list
            is_bool_indexes = True
            if len(indexes) != len(self._index):
                raise ValueError('boolean index list must be same size of existing index')
            bool_indexes = indexes
            indexes = list(compress(self._index, indexes))
        else:
            is_bool_indexes = False
            locations = [sorted_index(self._index, x) for x in indexes] if self._sort \
                else [self._index.index(x) for x in indexes]

        if all([isinstance(i, bool) for i in columns]):  # boolean list
            if len(columns) != len(self._columns):
                raise ValueError('boolean column list must be same size of existing columns')
            columns = list(compress(self._columns, columns))

        col_locations = [self._columns.index(x) for x in columns]
        data_dict = dict()

        for c in col_locations:
            data_dict[self._columns[c]] = list(compress(self._data[c], bool_indexes)) if is_bool_indexes \
                else [self._data[c][i] for i in locations]

        return DataFrame(data=data_dict, index=indexes, columns=columns, index_name=self._index_name,
                         sort=self._sort)