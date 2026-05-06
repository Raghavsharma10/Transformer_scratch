def spacings(self):
        """Computes the distances between neighboring crystal planes"""
        result_invsq = (self.reciprocal**2).sum(axis=0)
        result = np.zeros(3, float)
        for i in range(3):
            if result_invsq[i] > 0:
                result[i] = result_invsq[i]**(-0.5)
        return result