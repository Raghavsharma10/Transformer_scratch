def get_method(self, method_name, default=None):
        """
        Returns the contained method of the specified name, or `default` if
        not found.
        """
        for method in self.methods:
            if method.name == method_name:
                return method
        return default