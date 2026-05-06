def is_downstream_of(self, other):
        """
        return a boolean indicating whether this feature is downstream of
        `other` taking the strand of other into account
        """
        if self.chrom != other.chrom: return None
        if getattr(other, "strand", None) == "-":
            # other feature is on - strand, so this must have higher start
            return self.end <= other.start
        return self.start >= other.end