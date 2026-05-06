def add(self, item, position=5):
        """Add an item to the list unless it is already present.
        
        If the item is an expression, then a semicolon will be appended to it
        in the final compiled code.
        """
        if item in self.items:
            return
        self.items[item] = position
        self._add_dep(item)
        self.order = None
        self.changed(code_changed=True)