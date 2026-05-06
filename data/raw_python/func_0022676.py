def _edge_opposite_point(self, tri, i):
        """ Given a triangle, return the edge that is opposite point i.
        Vertexes are returned in the same orientation as in tri.
        """
        ind = tri.index(i)
        return (tri[(ind+1) % 3], tri[(ind+2) % 3])