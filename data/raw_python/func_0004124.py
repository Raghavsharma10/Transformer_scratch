def get_comparable_values(self):
        """Return a tupple of values representing the unicity of the object
        """
        return (int(self.order), str(self.label), str(self.name))