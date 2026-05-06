def distanceFromPoint(self, point):
        '''
        :param: point - Point subclass
        :return: float

        Distance from the line to the given point.
        '''
        # XXX planar distance, doesn't take into account z ?
        d = self.m
        n = (d.y * point.x) - (d.x * point.y) + self.A.cross(self.B)
        return abs(n / self.A.distance(self.B))