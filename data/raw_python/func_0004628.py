def _add_missing_rows(self, indexes):
        """
        Given a list of indexes, find all the indexes that are not currently in the Series and make a new row for
        that index by appending to the Series. This does not maintain sorted order for the index.

        :param indexes: list of indexes
        :return: nothing
        """
        new_indexes = [x for x in indexes if x not in self._index]
        for x in new_indexes:
            self._add_row(x)