def union(self, another_is):
        """
        Return the union between self and ``another_is``.

        Parameters
        ----------
        another_is : `IntervalSet`
            an IntervalSet object.
        Returns
        -------
        interval : `IntervalSet`
            the union of self with ``another_is``.
        """
        result = IntervalSet()
        if another_is.empty():
            result._intervals = self._intervals
        elif self.empty():
            result._intervals = another_is._intervals
        else:
            # res has no overlapping intervals
            result._intervals = IntervalSet.merge(self._intervals,
                                                  another_is._intervals,
                                                  lambda in_a, in_b: in_a or in_b)
        return result