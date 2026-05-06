def get_comparable_values(self):
        """Return a tupple of values representing the unicity of the object
        """
        return (not self.generic, int(self.code), str(self.message), str(self.description))