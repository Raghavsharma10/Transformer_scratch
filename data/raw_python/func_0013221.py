def get_child(self, name, attribs=None):
        """
        Returns the first child that matches the given name and
        attributes.
        """
        if name == '.':
            if attribs is None or len(attribs) == 0:
                return self
            if attribs == self.attribs:
                return self
        return self.child_index.get(nodehash(name, attribs))