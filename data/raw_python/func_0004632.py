def set_locations(self, locations, values):
        """
        For a list of locations set the values.

        :param locations: list of index locations
        :param values: list of values or a single value
        :return: nothing
        """

        indexes = [self._index[x] for x in locations]
        self.set(indexes, values)