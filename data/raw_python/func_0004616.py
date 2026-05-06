def get_location(self, location):
        """
        For an index location return a dict of the index and value. This is optimized for speed because
        it does not need to lookup the index location with a search. Also can accept relative indexing from the end of
        the SEries in standard python notation [-3, -2, -1]

        :param location: index location in standard python form of positive or negative number
        :return: dictionary
        """
        return {self.index_name: self._index[location], self.data_name: self._data[location]}