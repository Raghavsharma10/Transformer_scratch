def convex_hull(self):
        """Return an array of vertex indexes representing the convex hull.

        If faces have not been computed for this mesh, the function
        computes them.
        If no vertices or faces are specified, the function returns None.
        """
        if self._faces is None:
            if self._vertices is None:
                return None
            self.triangulate()
        return self._convex_hull