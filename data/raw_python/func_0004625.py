def set(self, indexes, values=None):
        """
        Given indexes will set a sub-set of the Series to the values provided. This method will direct to the below 
        methods based on what types are passed in for the indexes. If the indexes contains values not in the Series 
        then new rows or columns will be added.

        :param indexes: indexes value, list of indexes values, or a list of booleans.
        :param values: value or list of values to set. If a list then must be the same length as the indexes parameter.
        :return: nothing
        """
        if isinstance(indexes, (list, blist)):
            self.set_rows(indexes, values)
        else:
            self.set_cell(indexes, values)