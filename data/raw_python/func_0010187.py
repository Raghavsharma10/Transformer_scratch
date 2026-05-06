def rotate2d(self, theta, origin=None, axis='z', radians=False):
        '''
        :theta: float radians to rotate self around origin
        :origin: optional Point, defaults to 0,0,0

        Returns a Point rotated by :theta: around :origin:.
        '''

        origin = Point._convert(origin)

        delta = self - origin

        p = Point(origin)

        if not radians:
            theta = math.radians(theta)

        cosT = math.cos(theta)
        sinT = math.sin(theta)

        if axis == 'z':
            p.x += (cosT * delta.x) - (sinT * delta.y)
            p.y += (sinT * delta.x) + (cosT * delta.y)
            return p

        if axis == 'y':
            p.z += (cosT * delta.z) - (sinT * delta.x)
            p.x += (sinT * delta.z) + (cosT * delta.x)
            return p

        if axis == 'x':
            p.y += (cosT * delta.y) - (sinT * delta.z)
            p.z += (sinT * delta.y) + (cosT * delta.z)
            return p

        raise KeyError('unknown axis {}, expecting x, y or z'.format(axis))