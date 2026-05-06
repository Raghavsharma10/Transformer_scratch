def get_parameter(self, name):
        """Get a parameter."""
        default_value = "$%!)(INVALID)(!%$"
        value = self.lib.tdGetDeviceParameter(self.id, name, default_value)
        if value == default_value:
            raise AttributeError(name)
        return value