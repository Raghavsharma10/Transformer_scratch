def value_contains(self, value, attribute):
        """
        Determine if any of the items in the value list for the given
        attribute contain value.
        """
        for item in self[attribute]:
            if value in item:
                return True
        return False