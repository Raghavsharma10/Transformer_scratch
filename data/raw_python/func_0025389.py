def merge(self, other_roc):
        """
        Ingest the values of another DistributedROC object into this one and update the statistics inplace.

        Args:
            other_roc: another DistributedROC object.
        """
        if other_roc.thresholds.size == self.thresholds.size and np.all(other_roc.thresholds == self.thresholds):
            self.contingency_tables += other_roc.contingency_tables
        else:
            print("Input table thresholds do not match.")