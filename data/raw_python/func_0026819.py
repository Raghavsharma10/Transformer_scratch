def getData(self):
        """
        Return a list of normalized InputPoint objects
        for the contour drawn with this pen.
        """
        # organize the points into segments
        # 1. make sure there is an on curve
        haveOnCurve = False
        for point in self._points:
            if point.segmentType is not None:
                haveOnCurve = True
                break
        # 2. move the off curves to front of the list
        if haveOnCurve:
            _prepPointsForSegments(self._points)
        # 3. ignore double points on start and end
        firstPoint = self._points[0]
        lastPoint = self._points[-1]
        if firstPoint.segmentType is not None and lastPoint.segmentType is not None:
            if firstPoint.coordinates == lastPoint.coordinates:
                if (firstPoint.segmentType in ["line", "move"]):
                    del self._points[0]
                else:
                    raise AssertionError("Unhandled point type sequence")
        # done
        return self._points