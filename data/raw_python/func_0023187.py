def triangulate(self):
        """
        Triangulates the set of vertices and stores the triangles in faces and
        the convex hull in convex_hull.
        """
        npts = self._vertices.shape[0]
        if np.any(self._vertices[0] != self._vertices[1]):
            # start != end, so edges must wrap around to beginning.
            edges = np.empty((npts, 2), dtype=np.uint32)
            edges[:, 0] = np.arange(npts)
            edges[:, 1] = edges[:, 0] + 1
            edges[-1, 1] = 0
        else:
            # start == end; no wrapping required.
            edges = np.empty((npts-1, 2), dtype=np.uint32)
            edges[:, 0] = np.arange(npts)
            edges[:, 1] = edges[:, 0] + 1

        tri = Triangulation(self._vertices, edges)
        tri.triangulate()
        return tri.pts, tri.tris