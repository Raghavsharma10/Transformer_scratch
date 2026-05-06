def circumcenter(self):
        '''
        The intersection of the median perpendicular bisectors, Point.

        The center of the circumscribed circle, which is the circle that
        passes through all vertices of the triangle.

        https://en.wikipedia.org/wiki/Circumscribed_circle#Cartesian_coordinates_2

        BUG: only finds the circumcenter in the XY plane
        '''

        if self.isRight:
            return self.hypotenuse.midpoint

        if self.A.isOrigin:
            t = self
        else:
            # translate triangle to origin
            t = Triangle(self.A - self.A, self.B - self.A, self.C - self.A)
            # XXX translation would be easier by defining add and sub for points
            # t = self - self.A  

        if not t.A.isOrigin:
            raise ValueError('failed to translate {} to origin'.format(t))

        BmulC = t.B * t.C.yx

        d = 2 * (BmulC.x - BmulC.y)

        bSqSum = sum((t.B ** 2).xy)
        cSqSum = sum((t.C ** 2).xy)

        x = (((t.C.y * bSqSum) - (t.B.y * cSqSum)) / d) + self.A.x
        y = (((t.B.x * cSqSum) - (t.C.x * bSqSum)) / d) + self.A.y

        return Point(x, y)