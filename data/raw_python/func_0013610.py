def checkQueryRange(self, start, end):
        """
        Checks to ensure that the query range is valid within this reference.
        If not, raise ReferenceRangeErrorException.
        """
        condition = (
            (start < 0 or end > self.getLength()) or
            start > end or start == end)
        if condition:
            raise exceptions.ReferenceRangeErrorException(
                self.getId(), start, end)