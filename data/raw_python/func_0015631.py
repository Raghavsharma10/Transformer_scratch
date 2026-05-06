def child_get(self, child, *prop_names):
        """Returns a list of child property values for the given names."""

        return [self.child_get_property(child, name) for name in prop_names]