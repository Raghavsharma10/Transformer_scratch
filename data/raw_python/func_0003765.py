def yield_pair_gradients(self, index1, index2):
        """Yields pairs ((s'(r_ij), grad_i v(bar{r}_ij))"""
        strength = self.strengths[index1, index2]
        distance = self.distances[index1, index2]
        yield -6*strength*distance**(-7), np.zeros(3)