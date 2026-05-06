def get_tree_root(self):
        """ Returns the absolute root node of current tree structure."""
        root = self
        while root.up is not None:
            root = root.up
        return root