def circumcircleForTriangle(cls, triangle):
        '''
        :param: triangle - Triangle class
        :return: Circle class

        Returns the circle where every vertex in the input triangle is
        on the radius of that circle.

        '''

        if triangle.isRight:
            # circumcircle origin is the midpoint of the hypotenues
            o = triangle.hypotenuse.midpoint
            r = o.distance(triangle.A)
            return cls(o, r)

        # otherwise
        # 1. find the normals to two sides
        # 2. translate them to the midpoints of those two sides
        # 3. intersect those lines for center of circumcircle
        # 4. radius is distance from center to any vertex in the triangle

        abn = triangle.AB.normal
        abn += triangle.AB.midpoint

        acn = triangle.AC.normal
        acn += triangle.AC.midpoint

        o = abn.intersection(acn)
        r = o.distance(triangle.A)
        return cls(o, r)