def add_attribute(self, path, name, value):
        """
        Creates the given attribute and sets it to the given value.
        Returns the (new or existing) node to which the attribute was added.
        """
        node = self.add(path)
        node.attribs.append((name, value))
        return node