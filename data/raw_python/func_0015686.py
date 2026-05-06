def set_property(self, name, value):
        """set_property(property_name: str, value: object)

        Set property *property_name* to *value*.
        """

        if not hasattr(self.props, name):
            raise TypeError("Unknown property: %r" % name)
        setattr(self.props, name, value)