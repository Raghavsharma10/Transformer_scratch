def _adjacent_tri(self, edge, i):
        """
        Given a triangle formed by edge and i, return the triangle that shares
        edge. *i* may be either a point or the entire triangle.
        """
        if not np.isscalar(i):
            i = [x for x in i if x not in edge][0]

        try:
            pt1 = self._edges_lookup[edge]
            pt2 = self._edges_lookup[(edge[1], edge[0])]
        except KeyError:
            return None
            
        if pt1 == i:
            return (edge[1], edge[0], pt2)
        elif pt2 == i:
            return (edge[1], edge[0], pt1)
        else:
            raise RuntimeError("Edge %s and point %d do not form a triangle "
                               "in this mesh." % (edge, i))