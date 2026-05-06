def _medianindex(self,v):
        """
        find new position of vertex v according to adjacency in layer l+dir.
        position is given by the median value of adjacent positions.
        median heuristic is proven to achieve at most 3 times the minimum
        of crossings (while barycenter achieve in theory the order of |V|)
        """
        assert self.prevlayer()!=None
        N = self._neighbors(v)
        g=self.layout.grx
        pos = [g[x].pos for x in N]
        lp = len(pos)
        if lp==0: return []
        pos.sort()
        pos = pos[::self.layout.dirh]
        i,j = divmod(lp-1,2)
        return [pos[i]] if j==0 else [pos[i],pos[i+j]]