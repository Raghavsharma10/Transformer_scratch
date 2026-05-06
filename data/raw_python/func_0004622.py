def equality(self, indexes=None, value=None):
        """
        Math helper method. Given a column and optional indexes will return a list of booleans on the equality of the
        value for that index in the DataFrame to the value parameter.

        :param indexes: list of index values or list of booleans. If a list of booleans then the list must be the same\
        length as the DataFrame
        :param value: value to compare
        :return: list of booleans
        """
        indexes = [True] * len(self._index) if indexes is None else indexes
        compare_list = self.get_rows(indexes, as_list=True)
        return [x == value for x in compare_list]