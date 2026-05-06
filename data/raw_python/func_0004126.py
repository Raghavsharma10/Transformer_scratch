def get_comparable_values_for_ordering(self):
        """Return a tupple of values representing the unicity of the object
        """

        return (0 if self.position >= 0 else 1, int(self.position), str(self.name), str(self.description))