def apply(self, coordinates):
        """Apply this distortion to Cartesian coordinates"""
        for i in self.affected_atoms:
            coordinates[i] = self.transformation*coordinates[i]