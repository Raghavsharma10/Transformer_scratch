def pyname(self):
        """Name of the compiled module."""
        if self.pymodule.endswith('__init__'):
            return self.pymodule.split('.')[-2]
        else:
            return self.pymodule.split('.')[-1]