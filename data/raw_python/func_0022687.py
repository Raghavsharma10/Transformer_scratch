def _orientation(self, edge, point):
        """ Returns +1 if edge[0]->point is clockwise from edge[0]->edge[1], 
        -1 if counterclockwise, and 0 if parallel.
        """
        v1 = self.pts[point] - self.pts[edge[0]]
        v2 = self.pts[edge[1]] - self.pts[edge[0]]
        c = np.cross(v1, v2)  # positive if v1 is CW from v2
        return 1 if c > 0 else (-1 if c < 0 else 0)