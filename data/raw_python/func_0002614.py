def _cc(self):
        """
        implementation of the efficient bilayer cross counting by insert-sort
        (see Barth & Mutzel paper "Simple and Efficient Bilayer Cross Counting")
        """
        g=self.layout.grx
        P=[]
        for v in self:
            P.extend(sorted([g[x].pos for x in self._neighbors(v)]))
        # count inversions in P:
        s = []
        count = 0
        for i,p in enumerate(P):
            j = bisect(s,p)
            if j<i: count += (i-j)
            s.insert(j,p)
        return count