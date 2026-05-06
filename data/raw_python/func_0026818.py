def tValueForPoint(self, point):
        """
        get a t values for a given point
        required:
            the point must be a point on the curve.
            in an overlap cause the point will be an intersection points wich is alwasy a point on the curve
        """
        if self.segmentType == "curve":
            on1 = self.previousOnCurve
            off1 = self.points[0].coordinates
            off2 = self.points[1].coordinates
            on2 = self.points[2].coordinates
            return _tValueForPointOnCubicCurve(point, (on1, off1, off2, on2))
        elif self.segmentType == "line":
            return _tValueForPointOnLine(point, (self.previousOnCurve, self.points[0].coordinates))
        elif self.segmentType == "qcurve":
            raise NotImplementedError
        else:
            raise NotImplementedError