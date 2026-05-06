def yield_pair_energies(self, index1, index2):
        """Yields pairs ((s(r_ij), v(bar{r}_ij))"""
        d_1 = 1/self.distances[index1, index2]
        if self.charges is not None:
            c1 = self.charges[index1]
            c2 = self.charges[index2]
            yield c1*c2*d_1, 1
        if self.dipoles is not None:
            d_3 = d_1**3
            d_5 = d_1**5
            delta = self.deltas[index1, index2]
            p1 = self.dipoles[index1]
            p2 = self.dipoles[index2]
            yield d_3*np.dot(p1, p2), 1
            yield -3*d_5, np.dot(p1, delta)*np.dot(delta, p2)
            if self.charges is not None:
                yield c1*d_3, np.dot(p2, delta)
                yield c2*d_3, np.dot(p1, -delta)