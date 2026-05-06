def _coord_vertical_alignment(self):
        """performs vertical alignment according to current dirvh internal state.
        """
        dirh,dirv = self.dirh,self.dirv
        g = self.grx
        for l in self.layers[::-dirv]:
            if not l.prevlayer(): continue
            r=None
            for vk in l[::dirh]:
                for m in l._medianindex(vk):
                    # take the median node in dirv layer:
                    um = l.prevlayer()[m]
                    # if vk is "free" align it with um's root
                    if g[vk].align is vk:
                        if dirv==1: vpair = (vk,um)
                        else:       vpair = (um,vk)
                        # if vk<->um link is used for alignment
                        if (vpair not in self.conflicts) and \
                           (r==None or dirh*r<dirh*m):
                            g[um].align = vk
                            g[vk].root = g[um].root
                            g[vk].align = g[vk].root
                            r = m