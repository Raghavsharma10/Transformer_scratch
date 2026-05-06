def on_unselect(self, item, action):
        """Add an action to make when an object is unfocused."""
        if not isinstance(item, int):
            item = self.items.index(item)

        self._on_unselect[item] = action