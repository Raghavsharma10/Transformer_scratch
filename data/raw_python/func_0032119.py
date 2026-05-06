def items(self):
        """
        Loads the items this Installation refers to.
        """
        for id in self._items:
            yield self.store.getItemByID(int(id))