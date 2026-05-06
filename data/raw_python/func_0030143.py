def measuredim(self):
        """Return a MeasureDimension proxy, which wraps the partition to provide access to
        columns in terms of measures and dimensions"""

        if isinstance(self, PartitionProxy):
            return MeasureDimensionPartition(self._obj)
        else:
            return MeasureDimensionPartition(self)