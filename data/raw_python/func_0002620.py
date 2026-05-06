def setrank(self,v):
        """set rank value for vertex v and add it to the corresponding layer.
           The Layer is created if it is the first vertex with this rank.
        """
        assert self.dag
        r=max([self.grx[x].rank for x in v.N(-1)]+[-1])+1
        self.grx[v].rank=r
        # add it to its layer:
        try:
            self.layers[r].append(v)
        except IndexError:
            assert r==len(self.layers)
            self.layers.append(Layer([v]))