def get_configurable_by_name(self, name):
        """
        Returns the registered configurable with the specified name or ``None`` if no
        such configurator exists.
        """
        l = [c for c in self.configurables if c.name == name]
        if l:
            return l[0]