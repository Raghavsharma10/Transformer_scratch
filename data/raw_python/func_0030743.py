def delete_partitions(self, ds):
        """Fast delete of all of a datasets codes, columns, partitions and tables"""
        from ambry.orm import Partition

        ssq = self.session.query

        ssq(Process).filter(Process.d_vid == ds.vid).delete()
        ssq(Code).filter(Code.d_vid == ds.vid).delete()
        ssq(ColumnStat).filter(ColumnStat.d_vid == ds.vid).delete()
        ssq(Partition).filter(Partition.d_vid == ds.vid).delete()