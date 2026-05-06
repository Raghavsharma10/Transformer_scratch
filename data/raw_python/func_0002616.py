def draw(self,N=1.5):
        """compute every node coordinates after converging to optimal ordering by N
           rounds, and finally perform the edge routing.
        """
        while N>0.5:
            for (l,mvmt) in self.ordering_step():
                pass
            N = N-1
        if N>0:
            for (l,mvmt) in self.ordering_step(oneway=True):
                pass
        self.setxy()
        self.draw_edges()