def yield_pair_energies(self, index1, index2):
        """Yields pairs ((s(r_ij), v(bar{r}_ij))"""
        strength = self.strengths[index1, index2]
        distance = self.distances[index1, index2]
        yield strength*distance**(-6), 1