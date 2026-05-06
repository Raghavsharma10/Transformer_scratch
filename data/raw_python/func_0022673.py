def _edge_event(self, i, j):
        """
        Force edge (i, j) to be present in mesh. 
        This works by removing intersected triangles and filling holes up to
        the cutting edge.
        """
        front_index = self._front.index(i)
        
        #debug("  == edge event ==")
        front = self._front

        # First just see whether this edge is already present
        # (this is not in the published algorithm)
        if (i, j) in self._edges_lookup or (j, i) in self._edges_lookup:
            #debug("    already added.")
            return
        #debug("    Edge (%d,%d) not added yet. Do edge event. (%s - %s)" % 
        #      (i, j, pts[i], pts[j]))
        
        # traverse in two different modes:
        #  1. If cutting edge is below front, traverse through triangles. These
        #     must be removed and the resulting hole re-filled. (fig. 12)
        #  2. If cutting edge is above the front, then follow the front until 
        #     crossing under again. (fig. 13)
        # We must be able to switch back and forth between these 
        # modes (fig. 14)

        # Collect points that draw the open polygons on either side of the 
        # cutting edge. Note that our use of 'upper' and 'lower' is not strict;
        # in some cases the two may be swapped.
        upper_polygon = [i]
        lower_polygon = [i]
        
        # Keep track of which section of the front must be replaced
        # and with what it should be replaced
        front_holes = []  # contains indexes for sections of front to remove
        
        next_tri = None   # next triangle to cut (already set if in mode 1)
        last_edge = None  # or last triangle edge crossed (if in mode 1)
        
        # Which direction to traverse front
        front_dir = 1 if self.pts[j][0] > self.pts[i][0] else -1
                
        # Initialize search state
        if self._edge_below_front((i, j), front_index):
            mode = 1  # follow triangles
            tri = self._find_cut_triangle((i, j))
            last_edge = self._edge_opposite_point(tri, i)
            next_tri = self._adjacent_tri(last_edge, i)
            assert next_tri is not None
            self._remove_tri(*tri)
            # todo: does this work? can we count on last_edge to be clockwise
            # around point i?
            lower_polygon.append(last_edge[1])
            upper_polygon.append(last_edge[0])
        else:
            mode = 2  # follow front

        # Loop until we reach point j
        while True:
            #debug("  == edge_event loop: mode %d ==" % mode)
            #debug("      front_holes:", front_holes, front)
            #debug("      front_index:", front_index)
            #debug("      next_tri:", next_tri)
            #debug("      last_edge:", last_edge)
            #debug("      upper_polygon:", upper_polygon)
            #debug("      lower_polygon:", lower_polygon)
            #debug("      =====")
            if mode == 1:
                # crossing from one triangle into another
                if j in next_tri:
                    #debug("    -> hit endpoint!")
                    # reached endpoint! 
                    # update front / polygons
                    upper_polygon.append(j)
                    lower_polygon.append(j)
                    #debug("    Appended to upper_polygon:", upper_polygon)
                    #debug("    Appended to lower_polygon:", lower_polygon)
                    self._remove_tri(*next_tri)
                    break
                else:
                    # next triangle does not contain the end point; we will
                    # cut one of the two far edges.
                    tri_edges = self._edges_in_tri_except(next_tri, last_edge)
                    
                    # select the edge that is cut
                    last_edge = self._intersected_edge(tri_edges, (i, j))
                    #debug("    set last_edge to intersected edge:", last_edge)
                    last_tri = next_tri
                    next_tri = self._adjacent_tri(last_edge, last_tri)
                    #debug("    set next_tri:", next_tri)
                    self._remove_tri(*last_tri)

                    # Crossing an edge adds one point to one of the polygons
                    if lower_polygon[-1] == last_edge[0]:
                        upper_polygon.append(last_edge[1])
                        #debug("    Appended to upper_polygon:", upper_polygon)
                    elif lower_polygon[-1] == last_edge[1]:
                        upper_polygon.append(last_edge[0])
                        #debug("    Appended to upper_polygon:", upper_polygon)
                    elif upper_polygon[-1] == last_edge[0]:
                        lower_polygon.append(last_edge[1])
                        #debug("    Appended to lower_polygon:", lower_polygon)
                    elif upper_polygon[-1] == last_edge[1]:
                        lower_polygon.append(last_edge[0])
                        #debug("    Appended to lower_polygon:", lower_polygon)
                    else:
                        raise RuntimeError("Something went wrong..")
                    
                    # If we crossed the front, go to mode 2
                    x = self._edge_in_front(last_edge)
                    if x >= 0:  # crossing over front
                        #debug("    -> crossed over front, prepare for mode 2")
                        mode = 2
                        next_tri = None
                        #debug("    set next_tri: None")
                        
                        # where did we cross the front?
                        # nearest to new point
                        front_index = x + (1 if front_dir == -1 else 0)
                        #debug("    set front_index:", front_index)
                        
                        # Select the correct polygon to be lower_polygon
                        # (because mode 2 requires this). 
                        # We know that last_edge is in the front, and 
                        # front[front_index] is the point _above_ the front. 
                        # So if this point is currently the last element in
                        # lower_polygon, then the polys must be swapped.
                        if lower_polygon[-1] == front[front_index]:
                            tmp = lower_polygon, upper_polygon
                            upper_polygon, lower_polygon = tmp
                            #debug('    Swap upper/lower polygons')
                        else:
                            assert upper_polygon[-1] == front[front_index]
                        
                    else:
                        assert next_tri is not None
                
            else:  # mode == 2
                # At each iteration, we require:
                #   * front_index is the starting index of the edge _preceding_
                #     the edge that will be handled in this iteration
                #   * lower_polygon is the polygon to which points should be
                #     added while traversing the front
                
                front_index += front_dir
                #debug("    Increment front_index: %d" % front_index)
                next_edge = (front[front_index], front[front_index+front_dir])
                #debug("    Set next_edge: %s" % repr(next_edge))
                
                assert front_index >= 0
                if front[front_index] == j:
                    # found endpoint!
                    #debug("    -> hit endpoint!")
                    lower_polygon.append(j)
                    upper_polygon.append(j)
                    #debug("    Appended to upper_polygon:", upper_polygon)
                    #debug("    Appended to lower_polygon:", lower_polygon)
                    break

                # Add point to lower_polygon. 
                # The conditional is because there are cases where the 
                # point was already added if we just crossed from mode 1.
                if lower_polygon[-1] != front[front_index]:
                    lower_polygon.append(front[front_index])
                    #debug("    Appended to lower_polygon:", lower_polygon)

                front_holes.append(front_index)
                #debug("    Append to front_holes:", front_holes)

                if self._edges_intersect((i, j), next_edge):
                    # crossing over front into triangle
                    #debug("    -> crossed over front, prepare for mode 1")
                    mode = 1
                    
                    last_edge = next_edge
                    #debug("    Set last_edge:", last_edge)
                    
                    # we are crossing the front, so this edge only has one
                    # triangle. 
                    next_tri = self._tri_from_edge(last_edge)
                    #debug("    Set next_tri:", next_tri)
                    
                    upper_polygon.append(front[front_index+front_dir])
                    #debug("    Appended to upper_polygon:", upper_polygon)
                #else:
                    #debug("    -> did not cross front..")
        
        #debug("Finished edge_event:")
        #debug("  front_holes:", front_holes)
        #debug("  upper_polygon:", upper_polygon)
        #debug("  lower_polygon:", lower_polygon)

        # (iii) triangluate empty areas
        
        #debug("Filling edge_event polygons...")
        for polygon in [lower_polygon, upper_polygon]:
            dist = self._distances_from_line((i, j), polygon)
            #debug("Distances:", dist)
            while len(polygon) > 2:
                ind = np.argmax(dist)
                #debug("Next index: %d" % ind)
                self._add_tri(polygon[ind], polygon[ind-1],
                              polygon[ind+1], legal=False, 
                              source='edge_event')
                polygon.pop(ind)
                dist.pop(ind)

        #debug("Finished filling edge_event polygons.")
        
        # update front by removing points in the holes (places where front 
        # passes below the cut edge)
        front_holes.sort(reverse=True)
        for i in front_holes:
            front.pop(i)