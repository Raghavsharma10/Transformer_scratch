def radiansBetween(self, other):
        '''
        :param: other - Line subclass
        :return: float

        Returns the angle measured between two lines in radians
        with a range of [0, 2 * math.pi].

        '''
        # a dot b = |a||b| * cos(theta)
        # a dot b / |a||b| = cos(theta)
        # cos-1(a dot b / |a||b|) = theta

        # translate each line so that it passes through the origin and
        # produce a new point whose distance (magnitude) from the
        # origin is 1.
        #

        a = Point.unit(self.A, self.B)
        b = Point.unit(other.A, other.B)

        # in a perfect world, after unit: |A| = |B| = 1
        # which is a noop when dividing the dot product of A,B
        # but sometimes the lengths are different.
        #
        # let's just assume things are perfect and the lengths equal 1.

        return math.acos(a.dot(b))