def is_child(self, node):
        """Check if a node is a child of the current node

        Parameters
        ----------
        node : instance of Node
            The potential child.

        Returns
        -------
        child : bool
            Whether or not the node is a child.
        """
        if node in self.children:
            return True
        for c in self.children:
            if c.is_child(node):
                return True
        return False