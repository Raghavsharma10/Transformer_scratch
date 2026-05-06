def get_comparable_values(self):
        """Return a tupple of values representing the unicity of the object
        """
        return (not self.generic, str(self.name), str(self.description))