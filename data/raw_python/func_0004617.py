def get_locations(self, locations, as_list=False):
        """
        For list of locations return a Series or list of the values.

        :param locations: list of index locations
        :param as_list: True to return a list of values
        :return: Series or list
        """

        indexes = [self._index[x] for x in locations]
        return self.get(indexes, as_list)