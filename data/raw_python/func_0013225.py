def open(self, path):
        """
        Creates and enters the given node, regardless of whether it already
        exists.
        Returns the new node.
        """
        self.current.append(self.create(path))
        return self.current[-1]