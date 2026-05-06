def set_parameter(self, name, value):
        """Set a parameter."""
        self.lib.tdSetDeviceParameter(self.id, name, str(value))