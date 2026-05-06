def _neighbors(self,v):
        """
        neighbors refer to upper/lower adjacent nodes.
        Note that v.N() provides neighbors of v in the graph, while
        this method provides the Vertex and DummyVertex adjacent to v in the
        upper or lower layer (depending on layout.dirv state).
        """
        assert self.layout.dag
        dirv = self.layout.dirv
        grxv = self.layout.grx[v]
        try: #(cache)
            return grxv.nvs[dirv]
        except AttributeError:
            grxv.nvs={-1:v.N(-1),+1:v.N(+1)}
            if grxv.dummy: return grxv.nvs[dirv]
            # v is real, v.N are graph neigbors but we need layers neighbors
            for d in (-1,+1):
                tr=grxv.rank+d
                for i,x in enumerate(v.N(d)):
                    if self.layout.grx[x].rank==tr:continue
                    e=v.e_with(x)
                    dum = self.layout.ctrls[e][tr]
                    grxv.nvs[d][i]=dum
            return grxv.nvs[dirv]