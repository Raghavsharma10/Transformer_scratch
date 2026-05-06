def get_comparable_values(self):
        """Return a tupple of values representing the unicity of the object
        """
        return (int(self.major), int(self.minor), str(self.label), str(self.name))