def append_rows(self, indexes, values):
        """
        Appends values to the end of the data. Be very careful with this function as for sort DataFrames it will not 
        enforce sort order. Use this only for speed when needed, be careful.

        :param indexes: list of indexes to append
        :param values: list of values to append
        :return: nothing
        """

        # check that the values data is less than or equal to the length of the indexes
        if len(values) != len(indexes):
            raise ValueError('length of values is not equal to length of indexes')

        # check the indexes are not duplicates
        combined_index = self._index + indexes
        if len(set(combined_index)) != len(combined_index):
            raise IndexError('duplicate indexes in Series')

        # append index value
        self._index.extend(indexes)
        self._data.extend(values)