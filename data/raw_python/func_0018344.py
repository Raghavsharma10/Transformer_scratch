def on_select(self, item, action):
        """
        Add an action to make when an object is selected.
        Only one action can be stored this way.
        """

        if not isinstance(item, int):
            item = self.items.index(item)

        self._on_select[item] = action