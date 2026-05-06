def get_transformation(self, coordinates):
        """Construct a transformation object"""
        atom1, atom2 = self.hinge_atoms
        direction = coordinates[atom1] - coordinates[atom2]
        direction /= np.linalg.norm(direction)
        direction *= np.random.uniform(-self.max_amplitude, self.max_amplitude)
        result = Translation(direction)
        return result