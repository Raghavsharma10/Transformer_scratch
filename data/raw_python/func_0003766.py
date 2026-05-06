def yield_pair_energies(self, index1, index2):
        """Yields pairs ((s(r_ij), v(bar{r}_ij))"""
        A = self.As[index1, index2]
        B = self.Bs[index1, index2]
        distance = self.distances[index1, index2]
        yield A*np.exp(-B*distance), 1