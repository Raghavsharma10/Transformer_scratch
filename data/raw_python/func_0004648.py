def get(self, indexes=None, columns=None, as_list=False, as_dict=False):
        """
        Given indexes and columns will return a sub-set of the DataFrame. This method will direct to the below methods
        based on what types are passed in for the indexes and columns. The type of the return is determined by the
        types of the parameters.

        :param indexes: index value, list of index values, or a list of booleans. If None then all indexes are used
        :param columns: column name or list of column names. If None then all columns are used
        :param as_list: if True then return the values as a list, if False return a DataFrame. This is only used if
            the get is for a single column
        :param as_dict: if True then return the values as a dictionary, if False return a DataFrame. This is only used
            if the get is for a single row
        :return: either DataFrame, list, dict or single value. The return is a shallow copy
        """
        if (indexes is None) and (columns is not None) and (not isinstance(columns, (list, blist))):
            return self.get_entire_column(columns, as_list)

        if indexes is None:
            indexes = [True] * len(self._index)
        if columns is None:
            columns = [True] * len(self._columns)

        if isinstance(indexes, (list, blist)) and isinstance(columns, (list, blist)):
            return self.get_matrix(indexes, columns)
        elif isinstance(indexes, (list, blist)) and (not isinstance(columns, (list, blist))):
            return self.get_rows(indexes, columns, as_list)
        elif (not isinstance(indexes, (list, blist))) and isinstance(columns, (list, blist)):
            return self.get_columns(indexes, columns, as_dict)
        else:
            return self.get_cell(indexes, columns)