def get_columns(self, index, columns=None, as_dict=False):
        """
        For a single index and list of column names return a DataFrame of the values in that index as either a dict
        or a DataFrame

        :param index: single index value
        :param columns: list of column names
        :param as_dict: if True then return the result as a dictionary
        :return: DataFrame or dictionary
        """
        i = sorted_index(self._index, index) if self._sort else self._index.index(index)
        return self.get_location(i, columns, as_dict)