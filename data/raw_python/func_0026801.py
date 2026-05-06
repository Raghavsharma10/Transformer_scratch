def parameters(self):
        """Get dict with all set parameters."""
        parameters = {}
        for name in self.PARAMETERS:
            try:
                parameters[name] = self.get_parameter(name)
            except AttributeError:
                pass
        return parameters