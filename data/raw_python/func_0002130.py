def get_attrs(self):
        """Get the global attributes from underlying data set."""
        return FrozenOrderedDict((a, getattr(self.ds, a)) for a in self.ds.ncattrs())