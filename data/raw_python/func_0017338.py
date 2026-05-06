def unregister(self, name):
        """Unregister function by name.
        """
        try:
            name = name.name
        except AttributeError:
            pass
        return self.pop(name,None)