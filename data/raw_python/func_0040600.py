def union(self,other):
        """
        Return a new DiscreteSet with the union of the two sets, i.e.
        all elements that are in self or in other.

        :param DiscreteSet other: Set to unite with
        :rtype: DiscreteSet
        """
        if self.everything:
            return self
        elif other.everything:
            return other
        else:
            return DiscreteSet(self.elements.union(other.elements))