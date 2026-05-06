def iter_ancestors(self):
        """ 
        Iterates over the list of all ancestor nodes from 
        current node to the current tree root.
        """
        node = self
        while node.up is not None:
            yield node.up
            node = node.up