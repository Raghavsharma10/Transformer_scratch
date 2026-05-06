def measure(self, vid):
        """Return a measure, given its vid or another reference"""

        from ambry.orm import Column

        if isinstance(vid, PartitionColumn):
            return vid
        elif isinstance(vid, Column):
            return PartitionColumn(vid)
        else:
            return PartitionColumn(self.table.column(vid), self)