def hessian(self):
        """Compute the hessian of the energy"""
        result = np.zeros((self.numc, 3, self.numc, 3), float)
        for index1 in range(self.numc):
            for index2 in range(self.numc):
                result[index1, :, index2, :] = self.hessian_component(index1, index2)
        return result