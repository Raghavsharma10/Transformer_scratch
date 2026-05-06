def add_item(self, item):
        """Updates the list of items in the current transaction"""
        _idx = len(self.items)
        self.items.update({"item_" + str(_idx + 1): item})