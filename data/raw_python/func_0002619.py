def _rank_optimize(self):
        """optimize ranking by pushing long edges toward lower layers as much as possible.
        see other interersting network flow solver to minimize total edge length
        (http://jgaa.info/accepted/2005/EiglspergerSiebenhallerKaufmann2005.9.3.pdf)
        """
        assert self.dag
        for l in reversed(self.layers):
            for v in l:
                gv = self.grx[v]
                for x in v.N(-1):
                   if all((self.grx[y].rank>=gv.rank for y in x.N(+1))):
                        gx = self.grx[x]
                        self.layers[gx.rank].remove(x)
                        gx.rank = gv.rank-1
                        self.layers[gv.rank-1].append(x)