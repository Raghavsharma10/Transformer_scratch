def apply(self, coordinates):
        """Generate, apply and return a random manipulation"""
        transform = self.get_transformation(coordinates)
        result = MolecularDistortion(self.affected_atoms, transform)
        result.apply(coordinates)
        return result