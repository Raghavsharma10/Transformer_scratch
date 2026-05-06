def _meanvalueattr(self,v):
        """
        find new position of vertex v according to adjacency in prevlayer.
        position is given by the mean value of adjacent positions.
        experiments show that meanvalue heuristic performs better than median.
        """
        sug = self.layout
        if not self.prevlayer(): return sug.grx[v].bar
        bars = [sug.grx[x].bar for x in self._neighbors(v)]
        return sug.grx[v].bar if len(bars)==0 else float(sum(bars))/len(bars)