def itermerged(self):
        """Iterate over all headers, merging duplicate ones together."""
        for key in self:
            val = _dict_getitem(self, key)
            yield val[0], ', '.join(val[1:])