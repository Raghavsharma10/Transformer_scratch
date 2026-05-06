def is_disjoint(self,other):
        """
        Check whether two Intervals are disjoint.

        :param Interval other: The Interval to check disjointedness with.
        """
        if self.is_empty() or other.is_empty():
            return True

        if self.bounds[0] < other.bounds[0]:
            i1,i2 = self,other
        elif self.bounds[0] > other.bounds[0]:
            i2,i1 = self,other
        else:
            #coincident lower bounds
            if self.is_discrete() and not other.included[0]:
                return True
            elif other.is_discrete() and not self.included[0]:
                return True
            else:
                return False

        return not i2.bounds[0] in i1