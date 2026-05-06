def get_descendants(self, strategy="levelorder", is_leaf_fn=None):
        """ Returns a list of all (leaves and internal) descendant nodes."""
        return [n for n in self.iter_descendants(
            strategy=strategy, is_leaf_fn=is_leaf_fn)]