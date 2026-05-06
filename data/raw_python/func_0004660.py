def set(self, indexes=None, columns=None, values=None):
        """
        Given indexes and columns will set a sub-set of the DataFrame to the values provided. This method will direct
        to the below methods based on what types are passed in for the indexes and columns. If the indexes or columns
        contains values not in the DataFrame then new rows or columns will be added.

        :param indexes: indexes value, list of indexes values, or a list of booleans. If None then all indexes are used
        :param columns: columns name, if None then all columns are used. Currently can only handle a single column or\
        all columns
        :param values: value or list of values to set (index, column) to. If setting just a single row, then must be a\
        dict where the keys are the column names. If a list then must be the same length as the indexes parameter, if\
        indexes=None, then must be the same and length of DataFrame
        :return: nothing
        """
        if (indexes is not None) and (columns is not None):
            if isinstance(indexes, (list, blist)):
                self.set_column(indexes, columns, values)
            else:
                self.set_cell(indexes, columns, values)
        elif (indexes is not None) and (columns is None):
            self.set_row(indexes, values)
        elif (indexes is None) and (columns is not None):
            self.set_column(indexes, columns, values)
        else:
            raise ValueError('either or both of indexes or columns must be provided')