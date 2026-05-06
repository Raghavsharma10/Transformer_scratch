def random(cls, origin=None, radius=1):
        '''
        :origin: optional Point or point equivalent
        :radius: optional float, radius around origin
        :return: Point subclass

        Returns a point with random x, y and z coordinates bounded by
        the sphere defined by (origin,radius).

        If a sphere is not supplied, a unit sphere at the origin is
        used by default.
        '''

        p = cls(origin)

        r = random.uniform(0, radius)
        u = random.uniform(0, Two_Pi)
        v = random.uniform(-Half_Pi, Half_Pi)

        r_cosv = r * math.cos(v)

        p.x += r_cosv * math.cos(u)
        p.y += r_cosv * math.sin(u)
        p.z += radius * math.sin(v)

        return p