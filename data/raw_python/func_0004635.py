def delete(self, indexes):
        """
        Delete rows from the DataFrame

        :param indexes: either a list of values or list of booleans for the rows to delete
        :return: nothing
        """
        indexes = [indexes] if not isinstance(indexes, (list, blist)) else indexes
        if all([isinstance(i, bool) for i in indexes]):  # boolean list
            if len(indexes) != len(self._index):
                raise ValueError('boolean indexes list must be same size of existing indexes')
            indexes = [i for i, x in enumerate(indexes) if x]
        else:
            indexes = [sorted_index(self._index, x) for x in indexes] if self._sort \
                else [self._index.index(x) for x in indexes]
        indexes = sorted(indexes, reverse=True)  # need to sort and reverse list so deleting works
        for i in indexes:
            del self._data[i]
        # now remove from index
        for i in indexes:
            del self._index[i]