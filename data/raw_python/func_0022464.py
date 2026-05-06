def keys(self):
        """Return ids of all indexed documents."""
        result = []
        if self.fresh_index is not None:
            result += self.fresh_index.keys()
        if self.opt_index is not None:
            result += self.opt_index.keys()
        return result