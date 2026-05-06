def merge(self, other_rel):
        """
        Ingest another DistributedReliability and add its contents to the current object.

        Args:
            other_rel: a Distributed reliability object.
        """
        if other_rel.thresholds.size == self.thresholds.size and np.all(other_rel.thresholds == self.thresholds):
            self.frequencies += other_rel.frequencies
        else:
            print("Input table thresholds do not match.")