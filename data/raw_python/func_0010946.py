def equipable_classes(self):
        """ Returns a list of classes that _can_ use the item. """
        sitem = self._schema_item

        return [c for c in sitem.get("used_by_classes", self.equipped.keys()) if c]