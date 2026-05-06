def yield_pair_gradients(self, index1, index2):
        """Yields pairs ((s'(r_ij), grad_i v(bar{r}_ij))"""
        d_2 = 1/self.distances[index1, index2]**2
        if self.charges is not None:
            c1 = self.charges[index1]
            c2 = self.charges[index2]
            yield -c1*c2*d_2, np.zeros(3)
        if self.dipoles is not None:
            d_4 = d_2**2
            d_6 = d_2**3
            delta = self.deltas[index1, index2]
            p1 = self.dipoles[index1]
            p2 = self.dipoles[index2]
            yield -3*d_4*np.dot(p1, p2), np.zeros(3)
            yield 15*d_6, p1*np.dot(p2, delta) + p2*np.dot(p1, delta)
            if self.charges is not None:
                yield -3*c1*d_4, p2
                yield -3*c2*d_4, -p1