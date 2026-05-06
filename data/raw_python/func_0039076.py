def contains(self, pt):
        "Tests if the provided point is in the polygon."
        if isinstance(pt, Point):
            ptx, pty = pt.x, pt.y
            assert (self.spatialReference is None or \
                    self.spatialReference.wkid is None) or \
                    (pt.spatialReference is None or \
                     pt.spatialReference.wkid is None) or \
                   self.spatialReference == pt.spatialReference, \
                   "Spatial references do not match."
        else:
            ptx, pty = pt
        in_shape = False
        # Ported nearly line-for-line from the Javascript API
        for ring in self._json_rings:
            for idx in range(len(ring)):
                idxp1 = idx + 1
                if idxp1 >= len(ring):
                    idxp1 -= len(ring)
                pi, pj = ring[idx], ring[idxp1]
                # Divide-by-zero checks
                if (pi[1] == pj[1]) and pty >= min((pi[1], pj[1])):
                    if ptx >= max((pi[0], pj[0])):
                        in_shape = not in_shape
                elif (pi[0] == pj[0]) and pty >= min((pi[0], pj[0])):
                    if ptx >= max((pi[1], pj[1])):
                        in_shape = not in_shape
                elif (((pi[1] < pty and pj[1] >= pty) or 
                     (pj[1] < pty and pi[1] >= pty)) and 
                    (pi[0] + (pty - pi[1]) / 
                     (pj[1] - pi[1]) * (pj[0] - pi[0]) < ptx)):
                    in_shape = not in_shape
        return in_shape