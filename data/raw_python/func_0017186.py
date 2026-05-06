def iter_descendants(self, strategy="levelorder", is_leaf_fn=None):
        """ Returns an iterator over all descendant nodes."""
        for n in self.traverse(strategy=strategy, is_leaf_fn=is_leaf_fn):
            if n is not self:
                yield n