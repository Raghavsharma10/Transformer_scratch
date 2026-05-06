def yield_pair_gradients(self, index1, index2):
        """Yields pairs ((s'(r_ij), grad_i v(bar{r}_ij))"""
        A = self.As[index1, index2]
        B = self.Bs[index1, index2]
        distance = self.distances[index1, index2]
        yield -B*A*np.exp(-B*distance), np.zeros(3)