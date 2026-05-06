def _insert_missing_rows(self, indexes):
        """
        Given a list of indexes, find all the indexes that are not currently in the Series and make a new row for
        that index, inserting into the index. This requires the Series to be sorted=True

        :param indexes: list of indexes
        :return: nothing
        """
        new_indexes = [x for x in indexes if x not in self._index]
        for x in new_indexes:
            self._insert_row(bisect_left(self._index, x), x)