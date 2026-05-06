def intersection(self,other):
        """
        Return a new DiscreteSet with the intersection of the two sets, i.e.
        all elements that are in both self and other.

        :param DiscreteSet other: Set to intersect with
        :rtype: DiscreteSet
        """
        if self.everything:
            if other.everything:
                return DiscreteSet()
            else:
                return DiscreteSet(other.elements)
        else:
            if other.everything:
                return DiscreteSet(self.elements)
            else:
                return DiscreteSet(self.elements.intersection(other.elements))