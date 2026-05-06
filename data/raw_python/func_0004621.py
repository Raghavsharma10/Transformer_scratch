def select_index(self, compare, result='boolean'):
        """
        Finds the elements in the index that match the compare parameter and returns either a list of the values that
        match, of a boolean list the length of the index with True to each index that matches. If the indexes are
        tuples then the compare is a tuple where None in any field of the tuple will be treated as "*" and match all
        values.

        :param compare: value to compare as a singleton or tuple
        :param result: 'boolean' = returns a list of booleans, 'value' = returns a list of index values that match
        :return: list of booleans or values
        """
        if isinstance(compare, tuple):
            # this crazy list comprehension will match all the tuples in the list with None being an * wildcard
            booleans = [all([(compare[i] == w if compare[i] is not None else True) for i, w in enumerate(v)])
                        for x, v in enumerate(self._index)]
        else:
            booleans = [False] * len(self._index)
            if self._sort:
                booleans[sorted_index(self._index, compare)] = True
            else:
                booleans[self._index.index(compare)] = True
        if result == 'boolean':
            return booleans
        elif result == 'value':
            return list(compress(self._index, booleans))
        else:
            raise ValueError('only valid values for result parameter are: boolean or value.')