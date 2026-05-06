def get_comparable_values(self):
        """Return a tupple of values representing the unicity of the object
        """
        return (str(self.name), str(self.description), str(self.type), bool(self.optional), str(self.constraints) if isinstance(self, Constraintable) else "")