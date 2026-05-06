def _reductionForOffset(self, offset):
        """
        Calculate the total reduction for a given X axis offset.

        @param offset: The C{int} offset.
        @return: The total C{float} reduction that should be made for this
            offset.
        """
        reduction = 0
        for (thisOffset, thisReduction) in self._adjustments:
            if offset >= thisOffset:
                reduction += thisReduction
            else:
                break
        return reduction