def _fell_trees(self):
        """
        Removes trees from the forest according to the specified fell method.
        """
        if callable(self.fell_method):
            for tree in self.fell_method(list(self.trees)):
                self.trees.remove(tree)