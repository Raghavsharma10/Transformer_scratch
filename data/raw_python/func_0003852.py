def get_transformation(self, coordinates):
        """Construct a transformation object"""
        atom1, atom2, atom3, atom4 = self.hinge_atoms
        a = coordinates[atom1] - coordinates[atom2]
        a /= np.linalg.norm(a)
        b = coordinates[atom3] - coordinates[atom4]
        b /= np.linalg.norm(b)
        direction = 0.5*(a+b)
        direction *= np.random.uniform(-self.max_amplitude, self.max_amplitude)
        result = Translation(direction)
        return result