def is_upstream_of(self, other):
        """
        check if this is upstream of the `other` interval taking the strand of
        the other interval into account
        """
        if self.chrom != other.chrom: return None
        if getattr(other, "strand", None) == "+":
            return self.end <= other.start
        # other feature is on - strand, so this must have higher start
        return self.start >= other.end