def remove(self, item):
        """Remove an item from the list.
        """
        self.items.pop(item)
        self._remove_dep(item)
        self.order = None
        self.changed(code_changed=True)