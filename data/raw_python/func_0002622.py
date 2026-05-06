def setdummies(self,e):
        """creates and defines all needed dummy vertices for edge e.
        """
        v0,v1 = e.v
        r0,r1 = self.grx[v0].rank,self.grx[v1].rank
        if r0>r1:
            assert e in self.alt_e
            v0,v1 = v1,v0
            r0,r1 = r1,r0
        if (r1-r0)>1:
            # "dummy vertices" are stored in the edge ctrl dict,
            # keyed by their rank in layers.
            ctrl=self.ctrls[e]={}
            ctrl[r0]=v0
            ctrl[r1]=v1
            for r in xrange(r0+1,r1):
                self.dummyctrl(r,ctrl)