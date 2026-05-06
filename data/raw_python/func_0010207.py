def doesIntersect(self, other):
        '''
        :param: other - Circle class

        Returns True iff:
          self.center.distance(other.center) <= self.radius+other.radius
        '''

        otherType = type(other)

        if issubclass(otherType, Ellipse):
            distance = self.center.distance(other.center)
            radiisum = self.radius + other.radius
            return distance <= radiisum

        if issubclass(otherType, Line):
            raise NotImplementedError('doesIntersect,other is Line class')

        raise TypeError("unknown type '{t}'".format(t=otherType))