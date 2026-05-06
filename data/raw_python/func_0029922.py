def add_partition(self, p):
        """Add a partition identity as a child of a dataset identity."""

        if not self.partitions:
            self.partitions = {}

        self.partitions[p.vid] = p