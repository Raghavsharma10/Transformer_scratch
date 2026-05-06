def collapse(self):
        """
        Collapse the collection items into a single element (dict or list)

        :return: A new Collection instance with collapsed items
        :rtype: Collection
        """
        results = []

        if isinstance(self._items, dict):
            items = self._items.values()

        for values in items:
            if isinstance(values, Collection):
                values = values.all()

            results += values

        return Collection(results)