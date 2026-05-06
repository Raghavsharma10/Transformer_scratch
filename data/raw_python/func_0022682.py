def _find_edge_intersections(self):
        """
        Return a dictionary containing, for each edge in self.edges, a list
        of the positions at which the edge should be split.
        """
        edges = self.pts[self.edges]
        cuts = {}  # { edge: [(intercept, point), ...], ... }
        for i in range(edges.shape[0]-1):
            # intersection of edge i onto all others
            int1 = self._intersect_edge_arrays(edges[i:i+1], edges[i+1:])
            # intersection of all edges onto edge i
            int2 = self._intersect_edge_arrays(edges[i+1:], edges[i:i+1])
        
            # select for pairs that intersect
            err = np.geterr()
            np.seterr(divide='ignore', invalid='ignore')
            try:
                mask1 = (int1 >= 0) & (int1 <= 1)
                mask2 = (int2 >= 0) & (int2 <= 1)
                mask3 = mask1 & mask2  # all intersections
            finally:
                np.seterr(**err)
            
            # compute points of intersection
            inds = np.argwhere(mask3)[:, 0]
            if len(inds) == 0:
                continue
            h = int2[inds][:, np.newaxis]
            pts = (edges[i, 0][np.newaxis, :] * (1.0 - h) + 
                   edges[i, 1][np.newaxis, :] * h)
            
            # record for all edges the location of cut points
            edge_cuts = cuts.setdefault(i, [])
            for j, ind in enumerate(inds):
                if 0 < int2[ind] < 1:
                    edge_cuts.append((int2[ind], pts[j]))
                if 0 < int1[ind] < 1:
                    other_cuts = cuts.setdefault(ind+i+1, [])
                    other_cuts.append((int1[ind], pts[j]))
        
        # sort all cut lists by intercept, remove duplicates
        for k, v in cuts.items():
            v.sort(key=lambda x: x[0])
            for i in range(len(v)-2, -1, -1):
                if v[i][0] == v[i+1][0]:
                    v.pop(i+1)
        return cuts