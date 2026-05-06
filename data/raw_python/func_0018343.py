def select(self, item):
        """Select an arbitrary item, by possition or by reference."""
        self._on_unselect[self._selected]()
        self.selected().unfocus()

        if isinstance(item, int):
            self._selected = item % len(self)
        else:
            self._selected = self.items.index(item)

        self.selected().focus()
        self._on_select[self._selected]()