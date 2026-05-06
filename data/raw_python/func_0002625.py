def setxy(self):
        """computes all vertex coordinates (x,y) using
        an algorithm by Brandes & Kopf.
        """
        self._edge_inverter()
        self._detect_alignment_conflicts()
        inf = float('infinity')
        # initialize vertex coordinates attributes:
        for l in self.layers:
            for v in l:
                self.grx[v].root  = v
                self.grx[v].align = v
                self.grx[v].sink  = v
                self.grx[v].shift = inf
                self.grx[v].X     = None
                self.grx[v].x     = [0.0]*4
        curvh = self.dirvh # save current dirvh value
        for dirvh in xrange(4):
            self.dirvh = dirvh
            self._coord_vertical_alignment()
            self._coord_horizontal_compact()
        self.dirvh = curvh # restore it
        # vertical coordinate assigment of all nodes:
        Y = 0
        for l in self.layers:
            dY = max([v.view.h/2. for v in l])
            for v in l:
                vx = sorted(self.grx[v].x)
                # mean of the 2 medians out of the 4 x-coord computed above:
                avgm = (vx[1]+vx[2])/2.
                # final xy-coordinates :
                v.view.xy = (avgm,Y+dY)
            Y += 2*dY+self.yspace
        self._edge_inverter()