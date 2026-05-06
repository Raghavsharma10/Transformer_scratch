def triangulate(self):
        """Do the triangulation
        """
        self._initialize()
        
        pts = self.pts
        front = self._front
        
        ## Begin sweep (sec. 3.4)
        for i in range(3, pts.shape[0]):
            pi = pts[i]
            #debug("========== New point %d: %s ==========" % (i, pi))
            
            # First, triangulate from front to new point
            # This applies to both "point events" (3.4.1) 
            # and "edge events" (3.4.2).

            # get index along front that intersects pts[i]
            l = 0
            while pts[front[l+1], 0] <= pi[0]:
                l += 1
            pl = pts[front[l]]
            
            # "(i) middle case"
            if pi[0] > pl[0]:  
                #debug("  mid case")
                # Add a single triangle connecting pi,pl,pr
                self._add_tri(front[l], front[l+1], i)
                front.insert(l+1, i)
            # "(ii) left case"
            else:
                #debug("  left case")
                # Add triangles connecting pi,pl,ps and pi,pl,pr
                self._add_tri(front[l], front[l+1], i)
                self._add_tri(front[l-1], front[l], i)
                front[l] = i
            
            #debug(front)
                
            # Continue adding triangles to smooth out front
            # (heuristics shown in figs. 9, 10)
            #debug("Smoothing front...")
            for direction in -1, 1:
                while True:
                    # Find point connected to pi
                    ind0 = front.index(i)
                    ind1 = ind0 + direction
                    ind2 = ind1 + direction
                    if ind2 < 0 or ind2 >= len(front):
                        break
                    
                    # measure angle made with front
                    p1 = pts[front[ind1]]
                    p2 = pts[front[ind2]]
                    err = np.geterr()
                    np.seterr(invalid='ignore')
                    try:
                        angle = np.arccos(self._cosine(pi, p1, p2))
                    finally:
                        np.seterr(**err)
                    
                    # if angle is < pi/2, make new triangle
                    #debug("Smooth angle:", pi, p1, p2, angle)
                    if angle > np.pi/2. or np.isnan(angle):
                        break
                    
                    assert (i != front[ind1] and 
                            front[ind1] != front[ind2] and 
                            front[ind2] != i)
                    self._add_tri(i, front[ind1], front[ind2],
                                  source='smooth1')
                    front.pop(ind1)
            #debug("Finished smoothing front.")
            
            # "edge event" (sec. 3.4.2)
            # remove any triangles cut by completed edges and re-fill 
            # the holes.
            if i in self._tops:
                for j in self._bottoms[self._tops == i]:
                    # Make sure edge (j, i) is present in mesh
                    # because edge event may have created a new front list
                    self._edge_event(i, j)  
                    front = self._front 
                
        self._finalize()
        
        self.tris = np.array(list(self.tris.keys()), dtype=int)