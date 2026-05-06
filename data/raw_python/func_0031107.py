def coverage(self):
        """
        Get the fraction of a subject is matched by its set of reads.

        @return: The C{float} fraction of a subject matched by its reads.
        """
        if self._targetLength == 0:
            return 0.0

        coverage = 0
        for (intervalType, (start, end)) in self.walk():
            if intervalType == self.FULL:
                # Adjust start and end to ignore areas where the read falls
                # outside the target.
                coverage += (min(end, self._targetLength) - max(0, start))
        return float(coverage) / self._targetLength