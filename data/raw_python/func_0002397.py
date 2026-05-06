def lists(self, value, key=None):
        """
        Get a list with the values of a given key

        :rtype: list
        """
        results = map(lambda x: x[value], self._items)

        return list(results)