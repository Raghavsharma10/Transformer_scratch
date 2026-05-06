def isin(self, column, compare_list):
        """
        Returns a boolean list where each elements is whether that element in the column is in the compare_list.

        :param column: single column name, does not work for multiple columns
        :param compare_list: list of items to compare to
        :return: list of booleans
        """
        return [x in compare_list for x in self._data[self._columns.index(column)]]