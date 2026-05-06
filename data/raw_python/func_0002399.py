def unique(self):
        """
        Return only unique items from the collection list.

        :rtype: Collection
        """
        seen = set()
        seen_add = seen.add

        return Collection([x for x in self._items if not (x in seen or seen_add(x))])