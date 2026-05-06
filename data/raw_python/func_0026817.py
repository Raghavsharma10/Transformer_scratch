def split(self, tValues):
        """
        Split the segment according the t values
        """
        if self.segmentType == "curve":
            on1 = self.previousOnCurve
            off1 = self.points[0].coordinates
            off2 = self.points[1].coordinates
            on2 = self.points[2].coordinates
            return bezierTools.splitCubicAtT(on1, off1, off2, on2, *tValues)
        elif self.segmentType == "line":
            segments = []
            x1, y1 = self.previousOnCurve
            x2, y2 = self.points[0].coordinates
            dx = x2 - x1
            dy = y2 - y1
            pp = x1, y1
            for t in tValues:
                np = (x1+dx*t, y1+dy*t)
                segments.append([pp, np])
                pp = np
            segments.append([pp, (x2, y2)])
            return segments
        elif self.segmentType == "qcurve":
            raise NotImplementedError
        else:
            raise NotImplementedError