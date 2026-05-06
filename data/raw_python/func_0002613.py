def _crossings(self):
        """
        counts (inefficently but at least accurately) the number of
        crossing edges between layer l and l+dirv.
        P[i][j] counts the number of crossings from j-th edge of vertex i.
        The total count of crossings is the sum of flattened P:
        x = sum(sum(P,[]))
        """
        g=self.layout.grx
        P=[]
        for v in self:
            P.append([g[x].pos for x in self._neighbors(v)])
        for i,p in enumerate(P):
            candidates = sum(P[i+1:],[])
            for j,e in enumerate(p):
                p[j] = len(filter((lambda nx:nx<e), candidates))
            del candidates
        return P