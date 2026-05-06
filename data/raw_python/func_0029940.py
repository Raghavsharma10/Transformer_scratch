def partition_by_vid(self, ref):
        """A much faster way to get partitions, by vid only"""
        from ambry.orm import Partition

        p = self.session.query(Partition).filter(Partition.vid == str(ref)).first()
        if p:
            return self.wrap_partition(p)
        else:
            return None