def iteritems(self):
        """Iterate over all header lines, including duplicate ones."""
        for key in self:
            vals = _dict_getitem(self, key)
            for val in vals[1:]:
                yield vals[0], val