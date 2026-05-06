def available_styles(self):
        """ Returns a list of all styles defined for the item """
        styles = self._schema_item.get("styles", [])

        return list(map(operator.itemgetter("name"), styles))