def get(self, indexes, as_list=False):
        """
        Given indexes will return a sub-set of the Series. This method will direct to the specific methods
        based on what types are passed in for the indexes. The type of the return is determined by the
        types of the parameters.

        :param indexes: index value, list of index values, or a list of booleans.
        :param as_list: if True then return the values as a list, if False return a Series.
        :return: either Series, list, or single value. The return is a shallow copy
        """
        if isinstance(indexes, (list, blist)):
            return self.get_rows(indexes, as_list)
        else:
            return self.get_cell(indexes)