def get_property(self, name):
        """get_property(property_name: str) -> object

        Retrieves a property value.
        """

        if not hasattr(self.props, name):
            raise TypeError("Unknown property: %r" % name)
        return getattr(self.props, name)