def _intersect(start1, end1, start2, end2):
        """
        Returns the intersection of two intervals. Returns (None,None) if the intersection is empty.

        :param int start1: The start date of the first interval.
        :param int end1: The end date of the first interval.
        :param int start2: The start date of the second interval.
        :param int end2: The end date of the second interval.

        :rtype: tuple[int|None,int|None]
        """
        start = max(start1, start2)
        end = min(end1, end2)

        if start > end:
            return None, None

        return start, end