def ordering_step(self,oneway=False):
        """iterator that computes all vertices ordering in their layers
           (one layer after the other from top to bottom, to top again unless
           oneway is True).
        """
        self.dirv=-1
        crossings = 0
        for l in self.layers:
            mvmt = l.order()
            crossings += mvmt
            yield (l,mvmt)
        if oneway or (crossings == 0):
            return
        self.dirv=+1
        while l:
            mvmt = l.order()
            yield (l,mvmt)
            l = l.nextlayer()