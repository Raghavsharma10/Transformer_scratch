def radii_of_curvature(self):
        """The radius of curvature at each point on the Polymer primitive.

        Notes
        -----
        Each element of the returned list is the radius of curvature,
        at a point on the Polymer primitive. Element i is the radius
        of the circumcircle formed from indices [i-1, i, i+1] of the
        primitve. The first and final values are None.
        """
        rocs = []
        for i, _ in enumerate(self):
            if 0 < i < len(self) - 1:
                rocs.append(radius_of_circumcircle(
                    self[i - 1]['CA'], self[i]['CA'], self[i + 1]['CA']))
            else:
                rocs.append(None)
        return rocs