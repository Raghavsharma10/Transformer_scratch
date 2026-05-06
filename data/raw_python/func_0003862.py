def com(self):
        """the center of mass of the molecule"""
        return (self.coordinates*self.masses.reshape((-1,1))).sum(axis=0)/self.mass