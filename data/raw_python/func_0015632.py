def child_set(self, child, **kwargs):
        """Set a child properties on the given child to key/value pairs."""

        for name, value in kwargs.items():
            name = name.replace('_', '-')
            self.child_set_property(child, name, value)