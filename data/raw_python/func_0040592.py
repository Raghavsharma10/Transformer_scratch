def intersection(self,other):
        """
        Return a new Interval with the intersection of the two intervals,
        i.e.  all elements that are in both self and other.

        :param Interval other: Interval to intersect with
        :rtype: Interval
        """
        if self.bounds[0] < other.bounds[0]:
            i1,i2 = self,other
        else:
            i2,i1 = self,other

        if self.is_disjoint(other):
            return Interval((1,0),(True,True))

        bounds = [None,None]
        included = [None,None]
        #sets are not disjoint, so i2.bounds[0] in i1:
        bounds[0] = i2.bounds[0]
        included[0] = i2.included[0]

        if i2.bounds[1] in i1:
            bounds[1] = i2.bounds[1]
            included[1] = i2.included[1]
        else:
            bounds[1] = i1.bounds[1]
            included[1] = i1.included[1]

        return Interval(bounds,included)