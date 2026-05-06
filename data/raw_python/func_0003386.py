def withconfig(self, keysuffix):
        """
        Load configurations with this decorator
        """
        def decorator(cls):
            return self.loadconfig(keysuffix, cls)
        return decorator