def gradient_component(self, index1):
        """Compute the gradient of the energy for one atom"""
        result = np.zeros(3, float)
        for index2 in range(self.numc):
            if self.scaling[index1, index2] > 0:
                for (se, ve), (sg, vg) in zip(self.yield_pair_energies(index1, index2), self.yield_pair_gradients(index1, index2)):
                    result += (sg*self.directions[index1, index2]*ve + se*vg)*self.scaling[index1, index2]
        return result