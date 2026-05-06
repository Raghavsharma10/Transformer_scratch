def yield_pair_hessians(self, index1, index2):
        """Yields pairs ((s''(r_ij), grad_i (x) grad_i v(bar{r}_ij))"""
        d_1 = 1/self.distances[index1, index2]
        d_3 = d_1**3
        if self.charges is not None:
            c1 = self.charges[index1]
            c2 = self.charges[index2]
            yield 2*c1*c2*d_3, np.zeros((3, 3))
        if self.dipoles is not None:
            d_5 = d_1**5
            d_7 = d_1**7
            p1 = self.dipoles[index1]
            p2 = self.dipoles[index2]
            yield 12*d_5*np.dot(p1, p2), np.zeros((3, 3))
            yield -90*d_7, np.outer(p1, p2) + np.outer(p2, p1)
            if self.charges is not None:
                yield 12*c1*d_5, np.zeros((3, 3))
                yield 12*c2*d_5, np.zeros((3, 3))