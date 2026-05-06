def energy(self):
        """Compute the energy of the system"""
        result = 0.0
        for index1 in range(self.numc):
            for index2 in range(index1):
                if self.scaling[index1, index2] > 0:
                    for se, ve in self.yield_pair_energies(index1, index2):
                        result += se*ve*self.scaling[index1, index2]
        return result