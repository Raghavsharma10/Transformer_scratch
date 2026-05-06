def walk(self):
        """
        Get the non-overlapping read intervals that match the subject.

        @return: A generator that produces (TYPE, (START, END)) tuples, where
            where TYPE is either self.EMPTY or self.FULL and (START, STOP) is
            the interval. The endpoint (STOP) of the interval is not considered
            to be in the interval. I.e., the interval is really [START, STOP).
        """
        intervals = sorted(self._intervals)

        def nextFull():
            start, stop = intervals.pop(0)
            while intervals:
                if intervals[0][0] <= stop:
                    _, thisStop = intervals.pop(0)
                    if thisStop > stop:
                        stop = thisStop
                else:
                    break
            return (start, stop)

        if intervals:
            # If the first interval (read) starts after zero, yield an
            # initial empty section to get us to the first interval.
            if intervals[0][0] > 0:
                yield (self.EMPTY, (0, intervals[0][0]))

            while intervals:
                # Yield a full interval followed by an empty one (if there
                # is another interval pending).
                lastFull = nextFull()
                yield (self.FULL, lastFull)
                if intervals:
                    yield (self.EMPTY, (lastFull[1], intervals[0][0]))

            # Yield the final empty section, if any.
            if lastFull[1] < self._targetLength:
                yield (self.EMPTY, (lastFull[1], self._targetLength))

        else:
            yield (self.EMPTY, (0, self._targetLength))