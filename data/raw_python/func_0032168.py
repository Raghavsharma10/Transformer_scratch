def isPointInside(self, xp, yp):
        """Is the given point inside the polygon?

        Input:
        ------------
        xp, yp
            (floats) Coordinates of point in same units that
            array vertices are specified when object created.
        Returns:
        -----------
        **True** / **False**
        """

        point = np.array([xp, yp]).transpose()
        polygon = self.polygon
        numVert, numDim = polygon.shape

        #Subtract each point from the previous one.
        polyVec = np.roll(polygon, -1, 0) - polygon
        #Get the vector from each vertex to the given point
        pointVec = point - polygon

        crossProduct = np.cross(polyVec, pointVec)

        if np.all(crossProduct < 0) or np.all(crossProduct > 0):
            return True
        return False