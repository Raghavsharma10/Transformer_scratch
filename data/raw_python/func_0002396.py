def contains(self, key, value=None):
        """
        Determine if an element is in the collection

        :param key: The element
        :type key: int or str

        :param value: The value of the element
        :type value: mixed

        :return: Whether the element is in the collection
        :rtype: bool
        """
        if value is not None:
            if isinstance(self._items, list):
                return key in self._items and self._items[self._items.index(key)] == value

            return self._items.get(key) == value

        return key in self._items