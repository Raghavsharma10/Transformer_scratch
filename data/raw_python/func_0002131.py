def get_dimensions(self):
        """Get the dimensions from underlying data set."""
        return FrozenOrderedDict((k, len(v)) for k, v in self.ds.dimensions.items())