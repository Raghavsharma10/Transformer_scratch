def set_param(self, into, name):
        """
        Set parameter key, noting whether list value is "complex"
        """
        value, complex = self.getlist(name)
        if value is not None:
            into[name] = value
        return complex