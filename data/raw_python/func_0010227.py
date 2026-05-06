def random(cls, origin=None, radius=1):
        '''
        :origin: - optional Point subclass
        :radius: - optional float
        :return: Triangle

        Creates a triangle with random coordinates in the circle
        described by (origin,radius).  If origin is unspecified, (0,0)
        is assumed. If the radius is unspecified, 1.0 is assumed.

        '''
        # XXX no collinearity checks, possible to generate a
        #     line (not likely, just possible).
        #
        pts = set()
        while len(pts) < 3:
            p = Point.random(origin, radius)
            pts.add(p)
        return cls(pts)