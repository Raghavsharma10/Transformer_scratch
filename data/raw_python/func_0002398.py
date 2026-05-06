def map(self, callback):
        """
        Run a map over each of the item.

        :param callback: The map function
        :type callback: callable

        :rtype: Collection
        """
        if isinstance(self._items, dict):
            return Collection(list(map(callback, self._items.values())))

        return Collection(list(map(callback, self._items)))