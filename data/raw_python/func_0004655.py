def get_locations(self, locations, columns=None, **kwargs):
        """
        For list of locations and list of columns return a DataFrame of the values.

        :param locations: list of index locations
        :param columns: list of column names
        :param kwargs: will pass along these parameters to the get() method
        :return: DataFrame
        """

        indexes = [self._index[x] for x in locations]
        return self.get(indexes, columns, **kwargs)