def iter_leaf_names(self, is_leaf_fn=None):
        """Returns an iterator over the leaf names under this node."""
        for n in self.iter_leaves(is_leaf_fn=is_leaf_fn):
            yield n.name