def inertia_tensor(self):
        """the intertia tensor of the molecule"""
        result = np.zeros((3,3), float)
        for i in range(self.size):
            r = self.coordinates[i] - self.com
            # the diagonal term
            result.ravel()[::4] += self.masses[i]*(r**2).sum()
            # the outer product term
            result -= self.masses[i]*np.outer(r,r)
        return result