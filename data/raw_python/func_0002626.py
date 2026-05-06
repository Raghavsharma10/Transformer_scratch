def _detect_alignment_conflicts(self):
        """mark conflicts between edges:
        inner edges are edges between dummy nodes
        type 0 is regular crossing regular (or sharing vertex)
        type 1 is inner crossing regular (targeted crossings)
        type 2 is inner crossing inner (avoided by reduce_crossings phase)
        """
        curvh = self.dirvh # save current dirvh value
        self.dirvh=0
        self.conflicts = []
        for L in self.layers:
            last = len(L)-1
            prev = L.prevlayer()
            if not prev: continue
            k0=0
            k1_init=len(prev)-1
            l=0
            for l1,v in enumerate(L):
                if not self.grx[v].dummy: continue
                if l1==last or v.inner(-1):
                    k1=k1_init
                    if v.inner(-1):
                        k1=self.grx[v.N(-1)[-1]].pos
                    for vl in L[l:l1+1]:
                        for vk in L._neighbors(vl):
                            k = self.grx[vk].pos
                            if (k<k0 or k>k1):
                                self.conflicts.append((vk,vl))
                    l=l1+1
                    k0=k1
        self.dirvh = curvh