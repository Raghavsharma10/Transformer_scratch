def add(self, left_column, right_column, indexes=None):
        """
        Math helper method that adds element-wise two columns. If indexes are not None then will only perform the math
        on that sub-set of the columns.

        :param left_column: first column name
        :param right_column: second column name
        :param indexes: list of index values or list of booleans. If a list of booleans then the list must be the same\
        length as the DataFrame
        :return: list
        """
        left_list, right_list = self._get_lists(left_column, right_column, indexes)
        return [l + r for l, r in zip(left_list, right_list)]