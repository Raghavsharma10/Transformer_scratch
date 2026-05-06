def get_transformation(self, coordinates):
        """Construct a transformation object"""
        atom1, atom2, atom3 = self.hinge_atoms
        center = coordinates[atom2]
        a = coordinates[atom1] - coordinates[atom2]
        b = coordinates[atom3] - coordinates[atom2]
        axis = np.cross(a, b)
        norm = np.linalg.norm(axis)
        if norm < 1e-5:
            # We suppose that atom3 is part of the affected atoms
            axis = random_orthonormal(a)
        else:
            axis /= np.linalg.norm(axis)
        angle = np.random.uniform(-self.max_amplitude, self.max_amplitude)
        return Complete.about_axis(center, angle, axis)