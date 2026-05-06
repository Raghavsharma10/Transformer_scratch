def gradient(self):
        """Compute the gradient of the energy for all atoms"""
        result = np.zeros((self.numc, 3), float)
        for index1 in range(self.numc):
            result[index1] = self.gradient_component(index1)
        return result