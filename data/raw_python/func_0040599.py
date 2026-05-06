def difference(self,other):
        """
        Return a new DiscreteSet with the difference of the two sets, i.e.
        all elements that are in self but not in other.

        :param DiscreteSet other: Set to subtract
        :rtype: DiscreteSet
        :raises ValueError: if self is a set of everything
        """
        if self.everything:
            raise ValueError("Can not remove from everything")
        elif other.everything:
            return DiscreteSet([])
        else:
            return DiscreteSet(self.elements.difference(other.elements))