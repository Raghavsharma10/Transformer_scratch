def efield(self):
        """Compute the electrostatic potential at each atom due to other atoms"""
        result = np.zeros((self.numc,3), float)
        for index1 in range(self.numc):
            result[index1] = self.efield_component(index1)
        return result