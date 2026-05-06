def extract(self, disp):
        """
        Extraction operation
        @param displacement vector in index space
        @return the part of the domain that is exposed by the shift
        """
        res = copy.deepcopy(self)
        for i in range(self.ndims):
            d = disp[i]
            s = self.domain[i]
            if d > 0:
                res.domain[i] = slice(s.start, d)
            elif d < 0:
                res.domain[i] = slice(d, s.stop)
        return res