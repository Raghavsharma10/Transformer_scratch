def as_partition(self, **kwargs):
        """Return a PartitionName based on this name."""

        return PartitionName(**dict(list(self.dict.items()) + list(kwargs.items())))