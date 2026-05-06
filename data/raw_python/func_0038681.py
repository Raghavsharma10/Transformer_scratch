def intersect(self, start_point, end_point):
        """Intersect the line segment with the box return the first
        intersection point and normal vector pointing into space from
        the box side intersected.

        If the line does not intersect, or lies completely in one side
        of the box return (None, None)
        """
        sx, sy, sz = start_point
        ex, ey, ez = end_point
        p1x, p1y, p1z = self.point1
        p2x, p2y, p2z = self.point2
        start_inside = start_point in self
        end_inside = end_point in self
        if start_inside != end_inside:
            if (end_inside and sy > p2y) or (start_inside and ey >= p2y) and (ey != sy):
                # Test for itersection with bottom face
                t = (sy - p2y) / (ey - sy)
                ix = (ex - sx) * t + sx
                iy = p2y
                iz = (ez - sz) * t + sz
                if p1x <= ix <= p2x and p1z <= iz <= p2z:
                    return (ix, iy, iz), (0.0, (sy > p2y) * 2.0 - 1.0, 0.0)
            if (end_inside and sx < p1x) or (start_inside and ex <= p1x) and (ex != sx):
                # Test for itersection with left face
                t = (sx - p1x) / (ex - sx)
                ix = p1x
                iy = (ey - sy) * t + sy
                iz = (ez - sz) * t + sz
                if p1y <= iy <= p2y and p1z <= iz <= p2z:
                    return (ix, iy, iz), ((sx > p1x) * 2.0 - 1.0, 0.0, 0.0)
            if (end_inside and sy < p1y) or (start_inside and ey <= p1y) and (ey != sy):
                # Test for itersection with top face
                t = (sy - p1y) / (ey - sy)
                ix = (ex - sx) * t + sx
                iy = p1y
                iz = (ez - sz) * t + sz
                if p1x <= ix <= p2x and p1z <= iz <= p2z:
                    return (ix, iy, iz), (0.0, (sy > p1y) * 2.0 - 1.0, 0.0)
            if (end_inside and sx > p2x) or (start_inside and ex >= p2x) and (ex != sx):
                # Test for itersection with right face
                t = (sx - p2x) / (ex - sx)
                ix = p2x
                iy = (ey - sy) * t + sy
                iz = (ez - sz) * t + sz
                if p1y <= iy <= p2y and p1z <= iz <= p2z:
                    return (ix, iy, iz), ((sx > p2x) * 2.0 - 1.0, 0.0, 0.0)
            if (end_inside and sz > p2z) or (start_inside and ez >= p2z) and (ez != sz):
                # Test for itersection with far face
                t = (sz - p2z) / (ez - sz)
                ix = (ex - sx) * t + sx
                iy = (ey - sy) * t + sy
                iz = p2z
                if p1y <= iy <= p2y and p1x <= ix <= p2x:
                    return (ix, iy, iz), (0.0, 0.0, (sz > p2z) * 2.0 - 1.0)
            if (end_inside and sz < p1z) or (start_inside and ez <= p1z) and (ez != sz):
                # Test for itersection with near face
                t = (sz - p1z) / (ez - sz)
                ix = (ex - sx) * t + sx
                iy = (ey - sy) * t + sy
                iz = p1z
                if p1y <= iy <= p2y and p1x <= ix <= p2x:
                    return (ix, iy, iz), (0.0, 0.0, (sz > p1z) * 2.0 - 1.0)
        return None, None