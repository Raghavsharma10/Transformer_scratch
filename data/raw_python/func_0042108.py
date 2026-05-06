def set(self, name, value):
        """Set a variable before ``call()``."""
        if not hasattr(self, 'funcs'):
            raise StartupError('startup cannot be called again')
        self.variable_values[name] = value