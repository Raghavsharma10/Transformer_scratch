def find(self, start, end):
        """find all elements between (or overlapping) start and end"""
        if self.intervals and not end < self.intervals[0].start:
            overlapping = [i for i in self.intervals if i.end >= start
                                                    and i.start <= end]
        else:
            overlapping = []

        if self.left and start <= self.center:
            overlapping += self.left.find(start, end)

        if self.right and end >= self.center:
            overlapping += self.right.find(start, end)

        return overlapping