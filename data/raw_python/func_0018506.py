def distance(self, other_or_start=None, end=None, features=False):
        """
        check the distance between this an another interval
        Parameters
        ----------

        other_or_start : Interval or int
            either an integer or an Interval with a start attribute indicating
            the start of the interval

        end : int
            if `other_or_start` is an integer, this must be an integer
            indicating the end of the interval

        features : bool
            if True, the features, such as CDS, intron, etc. that this feature
            overlaps are returned.
        """
        if end is None:
            assert other_or_start.chrom == self.chrom

        other_start, other_end = get_start_end(other_or_start, end)

        if other_start > self.end:
            return other_start - self.end
        if self.start > other_end:
            return self.start - other_end
        return 0