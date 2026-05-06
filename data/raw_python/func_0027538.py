def items(self):
        """
        :return: a list of name/value attribute pairs sorted by attribute name.
        """
        sorted_keys = sorted(self.keys())
        return [(k, self[k]) for k in sorted_keys]